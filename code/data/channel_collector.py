import asyncio
import socket
import copy
import random
import csv

import os
import traceback



from aiohttp import ClientSession, TCPConnector, DummyCookieJar
from youtube_helpers import (
    BASE,
    BROWSE_ENDPOINT,
    PAYLOAD,
    USER_AGENT,
    get_channel_id,
    get_tab,
    get_continuation,
    tab_helpers,
)



####################### GLOBAL VARIABLES ############################
channel_lock = asyncio.Lock()
error_lock = asyncio.Lock()
error_file = open('collection_errors.txt', 'a', encoding = "utf-8")
#####################################################################



class Extractor():
    def __init__(self, name, filename, param, parse_func, features):
        self.name = name
        self.browse_param = param
        self.parse_func = parse_func
        self.features = features

        self.file = open(f'{filename}.csv', 'a', encoding = "utf-8")
        self.writer = csv.writer(self.file)
        self.lock = asyncio.Lock()

        # write header if it doesn't exist
        if os.path.getsize(f'{filename}.csv') == 0:
            self.writer.writerow(features)


    async def extract(self, session, channel_id, channel_link):
        # initial request to get load a "selected tab" (e.g. videos, playlists, etc.)
        payload = copy.deepcopy(PAYLOAD)
        payload['browseId'] = channel_id
        payload['params'] = self.browse_param
        async with session.post(
            BROWSE_ENDPOINT, headers = {'User-Agent': USER_AGENT},
            json = payload, timeout = 10
        ) as response:
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
                BROWSE_ENDPOINT, headers = {'User-Agent': USER_AGENT},
                json = payload, timeout = 10
            ) as response:
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
    async with session.get(
        channel_link, headers = {'User-Agent': USER_AGENT}, timeout = 10
    ) as response:
        # ignore 404 and just continue
        if response.status == 404:
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
                print('worker stopping, all channels collected')
                break
            channel_link = channels_left.pop()
        
        try:
            # get request to get channel id (and for realism)
            channel_id = await get_channel(channel_link, session)
            if channel_id is None:
                async with error_lock:
                    print(f'Channel {channel_link} not found, skipping')
                    error_file.write(f'Channel id not found for {channel_link}\n')
                continue

            # collect all data from channel using extractors
            for extractor in extractors:
                await extractor.extract(session, channel_id, channel_link)
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


async def main(num_workers):
    # extractors will be used to collect data from different tabs of a channel
    extractors = [Extractor(**tab) for tab in tab_helpers]

    # read in shuffled channels    
    channels = set()
    with open('shuffled_channels.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            channels.add(line.strip())

    # continue from previous run if possible
    collected = []
    with open('channels.csv', 'r', encoding = 'utf-8') as f:
        f.readline() # skip header
        for row in csv.reader(f):
            if len(row) == 0:
                continue
            collected.append(row[0]) ## the link
    collected_set = set(collected)
    print(f'found {len(collected_set)} many channels already collected out of {len(channels)}')

    # remove channels already read (but reprocess last 100 channels)
    channels = channels - collected_set
    channels = list(channels)
    print(f'will reprocess last 100 channels leaving {len(channels)} channels left')

    # shuffle channels (again to be safe)
    random.shuffle(channels)

    # add back last 100 channels after shuffling so they will be reprocessed first
    channels = channels + collected[-100:]

    # use a singular session to benefit from connection pooling
    # use ipv6 (helps with blocks)
    conn = TCPConnector(limit = None, family = socket.AF_INET6, force_close = True)
    async with ClientSession(
        base_url = BASE, connector = conn, cookie_jar = DummyCookieJar()
    ) as session:
        # start the workers
        print('starting workers now, try loading `channels.csv` (say with pandas) to monitor progress')
        try:
            await asyncio.gather(*[
                worker(channels, session, extractors) for _ in range(num_workers)
            ])
        except (KeyboardInterrupt, Exception):
            print(traceback.format_exc())
            
            for extractor in extractors:
                extractor.file.flush()
                extractor.file.close()



##################### TOP LEVEL CODE ##########################
# windows specific fix
if os.name == 'nt': 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

asyncio.run(main(num_workers = 50))
