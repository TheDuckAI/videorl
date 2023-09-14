import asyncio
import socket
import copy
import json
import re
import os
import traceback

from bs4 import BeautifulSoup as bs4

from aiohttp import ClientSession, TCPConnector, DummyCookieJar
from youtube_helpers import (
    BASE,
    BROWSE_ENDPOINT,
    PAYLOAD,
    USER_AGENT,
    csv_writer,
    parse_channel_info,
    parse_videos,
)



######################### GLOBAL VARIABLES #################################
write_lock = asyncio.Lock()
print_lock = asyncio.Lock()
channel_lock = asyncio.Lock()
error_lock = asyncio.Lock()

channel_file = open('channels.tsv', 'a', encoding = "utf-8")
channel_writer = csv_writer(channel_file, delimiter = '\t')

video_file = open('videos.tsv', 'a', encoding = "utf-8")
video_writer = csv_writer(video_file, delimiter = '\t')

error_file = open('collection_errors.txt', 'a', encoding = "utf-8")
############################################################################


async def collect_videos(
    channel_link, session, channel_writer = channel_writer, video_writer = video_writer
):
    async with session.get(
        channel_link + '/videos', headers = {'User-Agent': USER_AGENT}, timeout = 10
    ) as response:
        # ignore 404 and just continue
        if response.status == 404:
            return
        
        if response.ok is False:
            print('bad response: ', response.status, response.reason)
            return response.ok
    
        # find the channel data within the html
        html = await response.text()
        soup = bs4(html, 'html.parser')
        data_str = 'var ytInitialData = '
        channel_data = json.loads(
            soup(text = re.compile(data_str))[0].strip(data_str).strip(';')
        )
        channel_info = [channel_link] + parse_channel_info(channel_data)
        
        # ignore title-less or "Topic" channels
        if channel_info[1] is None or channel_info[1].endswith(" - Topic"):
            return
        
        video_rows, token = parse_videos(channel_data, channel_link)

        # some channels may not have videos
        if video_rows is None:
            return

        async with write_lock:
            channel_writer.writerow(channel_info)        
            video_writer.writerows(video_rows)

    # continue "scrolling" through channel's videos
    while token is not None:
        data = copy.deepcopy(PAYLOAD)
        data['continuation'] = token
        async with session.post(
            BROWSE_ENDPOINT, headers = {'User-Agent': USER_AGENT},
            json = data, timeout = 10
        ) as response:
            if response.ok is False:
                print('bad response: ', response.status, response.reason)
                return response.ok
            
            video_data = await response.json()
            video_rows, token = parse_videos(video_data, channel_link, is_continuation = True)
            async with write_lock:
                video_writer.writerows(video_rows)



async def worker(channels_left, session):
    # async with print_lock:
    #     print('worker started')
    while True:
        async with channel_lock:
            if len(channels_left) == 0:
                print('worker stopping, all channels collected')
                break
            channel_link = channels_left.pop()
        
        try:
             ok_response = await collect_videos(channel_link, session)
             if ok_response is False:
                async with print_lock:
                    print("Stopping worker due to bad response (try restarting)")
                return
        except asyncio.exceptions.TimeoutError:
            # retry the channel
            async with channel_lock:
                channels_left.append(channel_link)
            async with print_lock:
                print('Caught a timeout')
        except Exception as e:
            async with print_lock:
                error_file.write('Exception caught and written to error file')
            async with error_lock:
                error_file.write(f'Exception caught for {channel_link}')
                error_file.write(traceback.format_exc())
        
        # async with print_lock:
        #     print('collected all video from the channel', channel_link, end = "\t\t\t\r")
        


async def main(num_workers):
    # load all channels
    channels = list()
    with open('shuffled_channels.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            channels.append(line.strip())

    # continue from previous run or handle cold start
    with open('channels.tsv', 'r', encoding = "utf-8") as f:
        if f.readline() == '':
            channel_writer.writerow(
                ['link', 'name', 'description', 'subscribers', 'isFamilySafe', 'tags']
            )
        else:
            collected = set()
            while True:
                line = f.readline()
                if line == '':
                    break
                collected.add(line.strip())
            count = len(collected)
            print(f'found {count} many channels already collected out of {len(channels)}')

            # remove channels already read (but reprocess last 100 channels)
            count = count - 100
            if count > 0:
                channels = channels[:-count]
            print(f'will reprocess last 100 channels leaving {len(channels)} channels left')
    with open('videos.tsv', 'r', encoding = "utf-8") as f:
        if f.readline() == '':
            video_writer.writerow(
                ['channel_link', 'id', 'title', 'date', 'length', 'views']
            )

    # use a singular session to benefit from connection pooling
    # use ipv6 (helps with blocks)
    conn = TCPConnector(limit = None, family = socket.AF_INET6, force_close = True)
    async with ClientSession(
        base_url = BASE, connector = conn, cookie_jar = DummyCookieJar()
    ) as session:
        # start the workers
        print('starting workers now, run `wc -l channels.tsv` to monitor number of channels collected')
        await asyncio.gather(*[
            worker(channels, session) for _ in range(num_workers)
        ])



##################### TOP LEVEL CODE ##########################
# windows specific fix
if os.name == 'nt': 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


try:
    asyncio.run(main(num_workers = 50))
except (KeyboardInterrupt, Exception) as e:
    # print("\nfinal exception:", e)
    print(traceback.format_exc())
    video_file.close()
    channel_file.close()