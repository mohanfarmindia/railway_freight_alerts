from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from io import BytesIO
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from datetime import datetime, timedelta

# Set the path for Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\intel\OneDrive\Documents\FarmIndia_python\railway_frieght_charge_scrapper\tesseract.exe"

# C:\Users\intel\OneDrive\Documents\FarmIndia_python\railway_frieght_charge_scrapper\tesseract.exe
# Function to solve CAPTCHA with retry logic
def solve_captcha(driver, image_xpath):
    while True:
        try:
            captcha_element = driver.find_element(By.XPATH, image_xpath)
            captcha_image = captcha_element.screenshot_as_png
            captcha_image = Image.open(BytesIO(captcha_image))
            captcha_image = captcha_image.convert("L")
            captcha_image = captcha_image.filter(ImageFilter.MedianFilter(size=3))
            enhancer = ImageEnhance.Contrast(captcha_image)
            captcha_image = enhancer.enhance(2)
            captcha_text = pytesseract.image_to_string(captcha_image, config='--psm 6')
            cleaned_text = re.sub(r'[^A-Za-z0-9]', '', captcha_text)

            captcha_field = driver.find_element(By.XPATH, "//*[@id='captchaText']")
            captcha_field.clear()
            captcha_field.send_keys(cleaned_text)

            submit_button = driver.find_element(By.XPATH, "//*[@id='collapse1']/div[5]/button")
            submit_button.click()
            time.sleep(2)

            if not is_captcha_incorrect(driver, "//*[@id='errmsg']"):
                print("Captcha accepted, proceeding...")
                return True
        except NoSuchElementException:
            print("CAPTCHA element not found. Retrying...")
        except Exception as e:
            print(f"Error while solving CAPTCHA: {e}")
            time.sleep(1)

def is_captcha_incorrect(driver, error_xpath):
    try:
        error_message = driver.find_element(By.XPATH, error_xpath).text
        if "Captcha Code doesn't Match" in error_message:
            print("Detected Captcha error: Code doesn't match. Retrying...")
            return True
    except NoSuchElementException:
        pass
    return False

# Function to process a single region
def process_region(region_code):
    options = Options()
    options.add_argument("--headless=new")  # Run in headless mode for server
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    # Set up Chrome driver using WebDriverManager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    region_data_list = []
    try:
        url = "https://www.fois.indianrail.gov.in/FOISWebPortal/pages/FWP_ODROtsgDtls.jsp"
        driver.get(url)
        print(f"\nPage Loaded for region: {region_code}")
        wait = WebDriverWait(driver, 5)

        # Select the region from the dropdown
        outstanding_odr_option = wait.until(EC.presence_of_element_located((By.ID, "Zone")))
        outstanding_odr_option.click()
        outstanding_odr_option.send_keys(region_code)
        print(f"Selected '{region_code}' from the dropdown.")

        # Solve CAPTCHA
        captcha_image_xpath = "/html/body/div[4]/center/form/div/div[2]/div[4]/img[1]"
        if not solve_captcha(driver, captcha_image_xpath):
            print(f"Failed to solve Captcha for region {region_code}. Skipping...")
            return None

        print("Waiting for iframe to load...")
        data_div = wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='dataDiv']")))
        iframe = data_div.find_element(By.TAG_NAME, "iframe")
        driver.switch_to.frame(iframe)

        print("Waiting for the table to load...")
        table_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "body > div > table")))

        # Extract all rows without any filtering
        script = """
        let table = document.querySelector('body > div > table');
        let rows = Array.from(table.querySelectorAll('tr:not(:first-child):not(:nth-child(2))'));
        let data = rows.map(row => {
            let cells = Array.from(row.querySelectorAll('td'));
            return cells.map(cell => cell.innerText.trim());
        });
        return data;
        """
        all_rows_data = driver.execute_script(script)

        for row_data in all_rows_data:
            if any(row_data):
                print(f"Row: {row_data[:5]}...")
                region_data_list.append(row_data + [region_code])

        if not region_data_list:
            print(f"No data found for region {region_code}. Closing browser.")
            return None

        column_names = [
            "S.No.", "DVSN", "STTN FROM", "DEMAND NO.", "DEMAND DATE",
            "DEMAND TIME", "Expected loading date", "CNSR",
            "CNSG", "CMDT", "TT", "PC",
            "PBF", "VIA", "RAKE CMDT",
            "DSTN", "INDENTED TYPE",
            "INDENTED UNTS", "INDENTED 8W",
            "OTSG UNTS", "OTSG 8W",
            "SUPPLIED UNTS", "SUPPLIED TIME", "Region"
        ]

        df_region = pd.DataFrame(region_data_list, columns=column_names)
        return df_region

    except Exception as e:
        print(f"An error occurred while processing region {region_code}: {e}")
        return None

    finally:
        driver.quit()

# Region codes (hardcoded)
region_codes = [
    "CR", "DFCR", "EC", "ECO", "ER", "KR", "NC", "NE",
    "NF", "NPLR", "NR", "NW", "SC", "SE", "SEC", "SR", "SW", "WC", "WR"
]

# Process regions in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_region, region_codes), total=len(region_codes), desc="Processing Regions"))

# Save results
results_filtered = [df for df in results if df is not None]
if results_filtered:
    final_df = pd.concat(results_filtered, ignore_index=True)
    final_df.to_csv('filtered_output_combined_regions_daily.csv', index=False)
    print(final_df.head())
else:
    print("No data collected.")

print("Final file has been generated.")

import pandas as pd
from datetime import datetime, timedelta

# Load last run time
last_run_log_path = 'last_run.log'
try:
    with open(last_run_log_path, 'r') as f:
        last_run_time = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
except FileNotFoundError:
    last_run_time = datetime.now() - timedelta(days=1)  # Default to 24 hours ago if no log

current_time = datetime.now()
time_since_last_run = current_time - last_run_time

# Load the data
final_df = pd.read_csv('filtered_output_combined_regions_daily.csv')

# Convert DEMAND DATE and DEMAND TIME
final_df['DEMAND DATE'] = pd.to_datetime(final_df['DEMAND DATE'], format='%d-%m-%y', errors='coerce')
final_df['DEMAND TIME'] = final_df['DEMAND TIME'].astype(str)
final_df['DEMAND DATETIME'] = pd.to_datetime(final_df['DEMAND DATE'].dt.strftime("%d-%m-%y") + ' ' + final_df['DEMAND TIME'], format='%d-%m-%y %H:%M', errors='coerce')

# Filter based on RAKE CMDT values
filtered_df = final_df[final_df['RAKE CMDT'].isin(['M', 'DOC'])]

# Filter based on last run time
filtered_df = filtered_df[(filtered_df['DEMAND DATETIME'] >= last_run_time) & (filtered_df['DEMAND DATETIME'] <= current_time)]

#Number of indents
total_indents = len(filtered_df)

# Save current run time
with open(last_run_log_path, 'w') as f:
    f.write(current_time.strftime('%Y-%m-%d %H:%M:%S'))

# Load the mapping CSV files
division_mapping = pd.read_csv('division_mapping.csv')
station_names = pd.read_csv('station_names.csv')
consignee_names = pd.read_csv('consignee_names.csv')

# Create mapping dictionaries
division_dict = dict(zip(division_mapping['Short Form'], division_mapping['Full Form']))
station_dict = dict(zip(station_names['Short Form'], station_names['Full Form']))
consignee_dict = dict(zip(consignee_names['Short Form'], consignee_names['Full Form']))

# Map 'DVSN' column using division mapping
filtered_df['DVSN'] = filtered_df['DVSN'].map(division_dict).fillna(filtered_df['DVSN'])

# Map 'STTN FROM' and 'DSTN' columns using station names mapping
filtered_df['STTN FROM'] = filtered_df['STTN FROM'].map(station_dict).fillna(filtered_df['STTN FROM'])
filtered_df['DSTN'] = filtered_df['DSTN'].map(station_dict).fillna(filtered_df['DSTN'])

# Map 'CNSR' and 'CNSG' columns using consignee names mapping
filtered_df['CNSR'] = filtered_df['CNSR'].map(consignee_dict).fillna(filtered_df['CNSR'])
filtered_df['CNSG'] = filtered_df['CNSG'].map(consignee_dict).fillna(filtered_df['CNSG'])

# Display the updated DataFrame
print(filtered_df.head())

# Save the updated DataFrame if needed
filtered_df.to_csv('updated_filtered_df.csv', index=False)

filtered_df = pd.read_csv("updated_filtered_df.csv")
filtered_df

import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from telegram import Bot
import nest_asyncio
import asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Define file paths and bot details
pickle_file_path = 'data.pkl'
log_file_path = 'sent_rows_log.pkl'
last_run_log_path = 'last_run.log'  # Path to the last run log file
BOT_TOKEN = "7836500041:AAHOL2jJ8WGrRVeAnjJ3a354W6c6jgD22RU"
CHAT_IDS = {
    8147978368: "Mohan FarmIndia",
    499903657: "Mohan Personal",
    7967517419: "Rasheed",
    7507991236: "Vidish",
    8192726425: "Rishi",
    7725939583: "Kanhaiyalal Sir"
}

# Load master data
if os.path.exists(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        master_data = pickle.load(file)
        master_df = pd.DataFrame(master_data)
else:
    master_df = pd.DataFrame()

# Load sent rows log
if os.path.exists(log_file_path):
    with open(log_file_path, 'rb') as file:
        sent_rows_log = pickle.load(file)
        sent_rows_df = pd.DataFrame(sent_rows_log)
else:
    sent_rows_df = pd.DataFrame(columns=['S.No.', 'DEMAND DATE', 'DEMAND TIME'])

# Ensure required columns exist in sent_rows_df
for col in ['S.No.', 'DEMAND DATE', 'DEMAND TIME']:
    if col not in sent_rows_df.columns:
        sent_rows_df[col] = pd.Series(dtype='str')

# Load filtered data
filtered_df = pd.read_csv("updated_filtered_df.csv")

# Convert date columns to string type
if 'DEMAND DATE' in master_df.columns:
    master_df['DEMAND DATE'] = master_df['DEMAND DATE'].astype(str)
if 'DEMAND DATE' in filtered_df.columns:
    filtered_df['DEMAND DATE'] = filtered_df['DEMAND DATE'].astype(str)
if 'DEMAND DATE' in sent_rows_df.columns:
    sent_rows_df['DEMAND DATE'] = sent_rows_df['DEMAND DATE'].astype(str)

# Ensure consistency in data types
filtered_df['S.No.'] = filtered_df['S.No.'].astype(str)
sent_rows_df['S.No.'] = sent_rows_df['S.No.'].astype(str)

# Identify new rows
if not master_df.empty:
    common_columns = list(set(filtered_df.columns) & set(master_df.columns))
    for col in common_columns:
        filtered_df[col] = filtered_df[col].astype(str)
        master_df[col] = master_df[col].astype(str)
    new_rows = filtered_df.merge(master_df, how='left', indicator=True, on=common_columns).query('_merge == "left_only"').drop('_merge', axis=1)
else:
    new_rows = filtered_df

# Ensure DEMAND TIME is treated as a string
if 'DEMAND TIME' in new_rows.columns:
    new_rows['DEMAND TIME'] = new_rows['DEMAND TIME'].astype(str)
if 'DEMAND TIME' in sent_rows_df.columns:
    sent_rows_df['DEMAND TIME'] = sent_rows_df['DEMAND TIME'].astype(str)

# Remove already sent rows
new_rows_to_send = new_rows.merge(sent_rows_df, on=['S.No.', 'DEMAND DATE', 'DEMAND TIME'], how='left', indicator=True)\
                           .query('_merge == "left_only"').drop('_merge', axis=1)


# Load last run time
try:
    with open(last_run_log_path, 'r') as f:
        last_run_time = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
except FileNotFoundError:
    last_run_time = None
    
async def send_alerts(new_rows):
    bot = Bot(token=BOT_TOKEN)
    
    if new_rows.empty:
        message = "*Alert:* No new indents placed."
        for chat_id, name in CHAT_IDS.items():
            try:
                await bot.send_message(chat_id=chat_id, text=f"{name}, {message}", parse_mode='Markdown')
            except Exception as e:
                print(f"Error sending alert to {name}: {e}")
        return

    # Summary stats
    total_indents = len(filtered_df)
    count_M = filtered_df[filtered_df['RAKE CMDT'] == 'M'].shape[0]
    count_DOC = filtered_df[filtered_df['RAKE CMDT'] == 'DOC'].shape[0]
    unique_regions = ', '.join(filtered_df['Region'].dropna().unique().tolist())

    # Calculate time since last run in hours
    time_since_last_run_hours = time_since_last_run.total_seconds() / 3600
    summary_message = (
        f"*All India Indents Placed (Last {time_since_last_run_hours:.2f} hours):* {total_indents}\n"
        f"*Total Maize RAKES:* {count_M}\n"
        f"*Total DOC RAKES:* {count_DOC}\n"
        f"*Unique Regions:* {unique_regions}\n\n"
    )

    message = "*New Entries Alert:*\n\n" + summary_message

    # Group by DEMAND DATE
    grouped_rows = new_rows.groupby(['DEMAND DATE'])
    for demand_date, group in grouped_rows:
        date_str = demand_date if pd.notna(demand_date) else "Unknown"
        
        group_message = f"*Demand Date:* {date_str}\n"
        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            group_message += (
                f"{idx}. *From:* {row['STTN FROM']}\n"
                f"   *To:* {row['DSTN']}\n"
                f"   *CMDT:* {row['RAKE CMDT']}\n"
                f"   *CNSR:* {row.get('CNSR', 'N/A')}\n"
                f"   *CNSG:* {row.get('CNSG', 'N/A')}\n"
                f"   *DVSN:* {row.get('DVSN', 'N/A')}\n\n"
            )
        
        message += group_message + "\n"

    # Send alerts to each chat ID
    for chat_id, name in CHAT_IDS.items():
        try:
            await bot.send_message(chat_id=chat_id, text=f"{name}, {message}", parse_mode='Markdown')
        except Exception as e:
            print(f"Error sending alert to {name}: {e}")

    # Store sent rows in the log
    sent_rows_log = new_rows[['S.No.', 'DEMAND DATE', 'DEMAND TIME']].to_dict('records')
    with open(log_file_path, 'wb') as file:
        pickle.dump(sent_rows_log, file)

# Main Execution
async def main():
    await send_alerts(new_rows_to_send)

if __name__ == "__main__":
    asyncio.run(main())

    # Update master data
    if not filtered_df.empty:
        master_data = filtered_df.to_dict('records')
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(master_data, file)

    print("Alerts sent and master data updated.")
