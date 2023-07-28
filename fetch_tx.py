import numpy as np
from tqdm import tqdm
import json
import requests
import csv
import random
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import process_map 
import queue
import threading
import sys

API_KEYS = [
    "D2QM9JPD6UMK1XYCTK32SE9IRZCP8BF1AG",
    "AWCSTEYS9UFUUBFV93VMI515KM76GDD64M",
    "GQCU175SEDHUYD5XPBB612Z82A1FUG7AJE",
    "JSIA3K7IY1KIZMENRKNF68APJGX94HTGXF",
    "569PCB1XJH8YBFWPCPNPGGXXZZ1UVUHCTR",
    "BH636KQJA859VI1NMFZZZ2QBP8AK4SI2F3",
    "8R8Q31PKGGXMDQW1XZEZUASWEYB5GIQ4UU",
    "ERT6MCD5DURIGP4A7IKRWRVK27WUMG15UW",
    "9XJA73YK99XBGXN6ATNBXDAEQUREC2N8DD",
    "APNH5S9WU2JSQCYJKAR1IVKTW82V1EXMVJ"
    ]

def generate_random_number():
    return random.choice(range(10))

def fetch_transactions(address):
    transactions = {}
    i = generate_random_number()

    # Fetch normal transactions
    normal_url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=50&sort=asc&apikey={API_KEYS[i]}"
    normal_response = requests.get(normal_url).json()
    transactions['normal'] = normal_response['result']

    # # Fetch internal transactions
    # internal_url = f"https://api.etherscan.io/api?module=account&action=txlistinternal&address={address}&startblock=0&endblock=2702578&page=1&offset=50&sort=asc&apikey={API_KEYS[i]}"
    # internal_response = requests.get(internal_url).json()
    # transactions['internal'] = internal_response['result']

    # # Fetch ERC-20 transactions
    # erc20_url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={address}&contractaddress=0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2&page=1&offset=100&startblock=0&endblock=27025780&sort=asc&apikey={API_KEYS[i]}"
    # erc20_response = requests.get(erc20_url).json()
    # transactions['ERC20'] = erc20_response['result']

    # # Fetch ERC-721 transactions
    # erc721_url = f"https://api.etherscan.io/api?module=account&action=tokennfttx&address={address}&contractaddress=0x06012c8cf97bead5deae237070f9587f8e7a266d&page=1&offset=100&startblock=0&endblock=27025780&sort=asc&apikey={API_KEYS[i]}"
    # erc721_response = requests.get(erc721_url).json()
    # transactions['ERC721'] = erc721_response['result']

    return transactions

def worker(file_id, addresses, progress_queue):
    result = []
    for address in addresses:
        while True:
            try:
                fetched_data = fetch_transactions(address)
                fetched_data = {
                    address: fetched_data
                }
                result.append(fetched_data)
                break
            except Exception as err:
                print(f"Error occurred while fetching for address {address}. Retrying...")
        # Update the json file
        with open(f'temp_tx/user_transactions_{file_id}.json', 'w') as f:
            json.dump(result, f, indent=2)
        # Report progress
        progress_queue.put(file_id)


def progress_tracker(progress_queue, pbar_dict, total_tasks):
    finished_tasks = 0
    while finished_tasks < total_tasks:
        worker_id = progress_queue.get()
        pbar_dict[worker_id].update()
        finished_tasks += 1

def main():
    # Set the maximum field limit to the maximum possible
    csv.field_size_limit(sys.maxsize)
    # Read addresses from transactions.csv into a set
    processed_addresses = set()
    with open("dataset/user_transactions.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            processed_addresses.add(row[0])  # Address is in the first column

    # Read addresses from remaining_eoa_addresses.csv
    # and append only those that haven't been processed yet
    addresses = []
    with open('dataset/latest_remaining_eoa_addresses.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            address = row['address']
            if address not in processed_addresses:
                addresses.append(address)
    # print("num of processed addresses",len(processed_addresses))
    # print("num of remaining addresses",len(addresses))
    workers = 10
    split_data = np.array_split(addresses, workers)

    pbar_dict = {i: tqdm(total=len(split_data[i]), desc=f"Worker {i}") for i in range(workers)}
    progress_queue = queue.Queue()
    total_tasks = len(addresses)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers+1) as executor:  # Add one extra worker for the progress tracker
        executor.submit(progress_tracker, progress_queue, pbar_dict, total_tasks)  # Progress tracker
        for file_id, addresses in enumerate(split_data):
            executor.submit(worker, file_id, addresses, progress_queue)

    # Wait for all progress bars to finish
    for pbar in pbar_dict.values():
        pbar.close()

if __name__ == "__main__":
    main()
