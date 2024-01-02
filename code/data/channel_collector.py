import asyncio
import socket
import copy
import random
import csv
import os
import traceback


import pandas as pd
from aiohttp import ClientSession, TCPConnector, DummyCookieJar


from youtube_helpers import (
    BASE,
    BROWSE_ENDPOINT,
    PAYLOAD,
    HEADER,
    get_channel_id,
    get_tab,
    get_continuation,
    tab_helpers,
)



####################### GLOBAL VARIABLES ############################
channel_lock = asyncio.Lock()

error_lock = asyncio.Lock()
error_file = open('collection_errors.txt', 'a', encoding = "utf-8")

completion_lock = asyncio.Lock()
completion_file = open('fully_collected.txt', 'a', encoding = "utf-8")

request_lock = asyncio.Lock()
request_count = 0
#####################################################################



class Extractor():
    def __init__(self, name, filename, param, parse_func, features):
        self.name = name
        self.browse_param = param
        self.parse_func = parse_func
        self.features = features

        self.file_number = 1
        self.filename = filename
        self.open_new_file()

        os.makedirs(self.filename, exist_ok = True)

        self.lock = asyncio.Lock()

    def open_new_file(self):
        # Close the current file if it's open
        if hasattr(self, 'file'):
            self.file.close()

        # Open a new file with the current number
        self.file = open(f'{self.filename}.csv', 'a', encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Write header if it doesn't exist
        if os.path.getsize(f'{self.filename}.csv') == 0:
            self.writer.writerow(self.features)

        # Increment the file number for next time
        self.file_number += 1

    def convert_to_parquet_if_large(self, threshold_gb = 1, tolerance = 0.2, force = False):
        # Get the size of the file in GB
        file_size_gb = os.path.getsize(f'{self.filename}.csv') / (2**30)

        # Check if the file size is within 20% of the threshold
        if threshold_gb * (1 - tolerance) <= file_size_gb or force:
            # Convert the CSV file to Parquet
            df = pd.read_csv(f'{self.filename}.csv')
            
            # put file in a directory
            df.to_parquet(os.path.join(self.filename, f'{self.filename}_{self.file_number - 1}.parquet'))

            # Delete the CSV file
            self.file.flush()
            self.file.close()
            os.remove(f'{self.filename}.csv')

            # Open a new file for next time
            self.open_new_file()


    async def extract(self, session, channel_id, channel_link):
        global request_count
        
        # initial request to get load a "selected tab" (e.g. videos, playlists, etc.)
        payload = copy.deepcopy(PAYLOAD)
        payload['browseId'] = channel_id
        payload['params'] = self.browse_param
        async with session.post(
            BROWSE_ENDPOINT, headers = HEADER,
            json = payload, timeout = 10
        ) as response:
            async with request_lock:
                request_count += 1
            
            if response.ok is False:
                async with error_lock:
                    print('bad response: ', response.status, response.reason)
                raise BadResponseException()
            
            data = await response.json()
            tab = get_tab(data, self.name)

            # tab may not exist (e.g. if channel has no playlists)
            if tab is None:
                return
            
            rows, token = self.parse_func(
                channel_link, channel_id = channel_id,
                tab = tab, response = data
            )
            
            if rows is not None:
                async with self.lock:
                    self.writer.writerows(rows)

        # continue through continuation if it exists
        while token is not None:
            payload = copy.deepcopy(PAYLOAD)
            payload['continuation'] = token
            async with session.post(
                BROWSE_ENDPOINT, headers = HEADER,
                json = payload, timeout = 10
            ) as response:
                async with request_lock:
                    request_count += 1

                if response.ok is False:
                    async with error_lock:
                        print('bad response: ', response.status, response.reason)
                    raise BadResponseException()
                
                data = await response.json()
                continuation = get_continuation(data)

                if continuation is None:
                    return

                rows, token = self.parse_func(channel_link, continuation = continuation)

                if rows is not None:
                    async with self.lock:
                        self.writer.writerows(rows)


class BadResponseException(Exception):
    pass


########################## FUNCTION DECLARATIONS ###########################
async def get_channel(channel_link, session, error_lock = error_lock):
    global request_count

    async with session.get(
        channel_link, headers = HEADER,
        allow_redirects = False, timeout = 10
    ) as response:
        async with request_lock:
            request_count += 1

        # ignore 404 and just continue
        if response.status == 404 or response.status == 303:
            return None

        if response.ok is False:
            async with error_lock:
                print('bad response: ', response.status, response.reason)
            raise BadResponseException()
        
        return get_channel_id(await response.text())


async def worker(channels_left, session, extractors,
                 channel_lock = channel_lock, error_lock = error_lock):
    while True:
        async with channel_lock:
            if len(channels_left) == 0:
                break
            channel_link = channels_left.pop()
        
        try:
            # get request to get channel id (and for realism)
            channel_id = await get_channel(channel_link, session)
            if channel_id is None:
                # 404 or 303 and thus ignore
                # parsing errors will raise an assert instead
                continue

            # collect all data from channel using extractors
            for extractor in extractors:
                await extractor.extract(session, channel_id, channel_link)

            # write to completion file
            async with completion_lock:
                completion_file.write(channel_link + '\n')
        except BadResponseException:
            async with error_lock:
                print("Stopping worker due to bad response (try restarting)")
            break
        except asyncio.exceptions.TimeoutError:
            # retry the channel
            async with channel_lock:
                channels_left.append(channel_link)
            async with error_lock:
                print('Caught a timeout')
        except Exception:
            async with error_lock:
                print(f'Exception caught for {channel_link} and written to error file')
            async with error_lock:
                error_file.write(f'Exception caught for {channel_link}\n')
                error_file.write(traceback.format_exc())
                error_file.flush()


async def launch_workers(channels, extractors, num_workers = 1, block_size = 100):
    # use a singular session to benefit from connection pooling
    # use ipv6 (helps with blocks)
    while len(channels) > 0:
        # get a block of channels
        block = [channels.pop() for _ in range(min(block_size, len(channels)))]
        conn = TCPConnector(
            limit = None, family = socket.AF_INET6, force_close = True, ttl_dns_cache = 300
        )
        async with ClientSession(
            base_url = BASE, connector = conn, cookie_jar = DummyCookieJar()
        ) as session:
            # start the workers
            await asyncio.gather(*[
                worker(block, session, extractors) for _ in range(num_workers)
            ])
        # connection will close here

        print(f'finished block of channels, {len(channels)} left, num requests made: {request_count}', end = '\r')

        # if len(channels) > 0:
        #     # force compress extractors
        for extractor in extractors:
            extractor.convert_to_parquet_if_large()


async def main(num_workers, block_size):
    # extractors will be used to collect data from different tabs of a channel
    extractors = [Extractor(**tab) for tab in tab_helpers]

    # read in shuffled channels    
    channels = set()
    with open('shuffled_channels.txt', 'r', encoding = "ISO-8859-1") as f:
        for line in f:
            channels.add(line.strip())

    # continue from previous run if possible
    collected = []
    with open('fully_collected.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            collected.append(line.strip())
    collected_set = set(collected)
    print(f'found {len(collected_set)} many channels already collected out of {len(channels)}')

    # remove channels already read
    channels = channels - collected_set
    channels = list(channels)
    print(f'continuing with {len(channels)} many channels, now shuffling...')

    # shuffle channels (again to be safe)
    random.shuffle(channels)

    print(f'starting {num_workers} workers now, press ctrl-c to stop')
    try:
        await launch_workers(
            channels, extractors, num_workers = num_workers, block_size = block_size
        )
    except (KeyboardInterrupt, Exception):
        print(traceback.format_exc())

    print('number of requests made:', request_count)

    # clean up
    completion_file.flush()
    completion_file.close()
    for extractor in extractors:
        extractor.file.flush()
        extractor.convert_to_parquet_if_large(force = True)
        extractor.file.close()



##################### TOP LEVEL CODE ##########################
# windows specific fix
if os.name == 'nt': 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == '__main__':
    asyncio.run(main(num_workers = 30, block_size = 500))
