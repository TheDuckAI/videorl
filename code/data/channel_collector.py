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



# windows specific fix
if os.name == 'nt': 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

write_lock = asyncio.Lock()
print_lock = asyncio.Lock()
channel_lock = asyncio.Lock()

channel_file = open('channels.tsv', 'a', encoding = "utf-8")
channel_writer = csv_writer(channel_file, delimiter = '\t')

video_file = open('videos.tsv', 'a', encoding = "utf-8")
video_writer = csv_writer(video_file, delimiter = '\t')



async def collect_videos(
    channel_link, channel_writer = channel_writer, video_writer = video_writer
):
    # use ipv6 (helps with blocks) and leave concurrency to parallel connections
    conn = TCPConnector(limit = 1, family = socket.AF_INET6, force_close = True)
    async with ClientSession(
        base_url = BASE, connector = conn, cookie_jar = DummyCookieJar()
    ) as session:
        async with session.get(
            channel_link + '/videos', headers = {'User-Agent': USER_AGENT}, timeout = 5
        ) as response:
            # ignore 404 and just continue
            if response.status == 404:
                return
            
            if response.ok is False:
                print('bad response text', await response.text())
                return response.ok
        
            # find the channel data within the html
            html = await response.text()
            soup = bs4(html, 'html.parser')
            data_str = 'var ytInitialData = '
            channel_data = json.loads(
                soup(text = re.compile(data_str))[0].strip(data_str).strip(';')
            )
            async with write_lock:
                channel_info = [channel_link] + parse_channel_info(channel_data)
            
                # ignore "Topic" channels
                if channel_info[1].endswith(" - Topic"):
                    return
                
                channel_writer.writerow(channel_info)
                video_rows, token = parse_videos(channel_data)

                # some channels may not have videos
                if video_rows is None:
                    return
        
                video_writer.writerows(video_rows)

        # continue "scrolling" through channel's videos
        while token is not None:
            data = copy.deepcopy(PAYLOAD)
            data['continuation'] = token
            async with session.post(
                BROWSE_ENDPOINT, headers = {'User-Agent': USER_AGENT},
                json = data, timeout = 5
            ) as response:
                if response.ok is False:
                    print('bad response text', await response.text())
                    return response.ok
                
                video_data = await response.json()
                video_rows, token = parse_videos(video_data, is_continuation = True)
                async with write_lock:
                    video_writer.writerows(video_rows)



async def worker(channels_left):
    async with print_lock:
        print('worker started')

    while True:
        async with channel_lock:
            if len(channels_left) == 0:
                print('worker stopping, all channels collected')
                break
            channel_link = channels_left.pop()
        
        try:
             ok_response = await collect_videos(channel_link)
             if ok_response is False:
                async with print_lock:
                    print("bad response, stopping worker (try restarting)")
                return
        except Exception as e:
            async with print_lock:
                print(f'Exception caught for {channel_link}:', e)
        
        async with print_lock:
            print('collected all video from the channel', channel_link, end = "\t\t\t\r")
        


async def main(num_workers):
    # load all channels
    channels = list()
    with open('channels.txt', 'r') as f:
        for line in f:
            channels.append(line.strip())

    # continue from previous run or handle cold start
    with open('channels.tsv', 'r', encoding = "utf-8") as f:
        if f.readline() == '':
            channel_writer.writerow(
                ['link', 'name', 'description', 'subscribers', 'isFamilySafe', 'tags']
            )
        else:
            count = 0
            while True:
                line = f.readline()
                if line == '':
                    break
                count += 1
            print(f'found {count} many channels already collected out of {len(channels)}')

            # remove channels already read (but reprocess last 100 channels)
            count = count - 100
            if count > 0:
                channels = channels[:-count]
            print(f'will reprocess last 100 channels leaving {len(channels)} channels left')

    with open('videos.tsv', 'r', encoding = "utf-8") as f:
        if f.readline() == '':
            video_writer.writerow(
                ['id', 'title', 'date', 'length', 'views']
            )

    # start the workers
    await asyncio.gather(*[
        worker(channels) for _ in range(num_workers)
    ])

try:
    asyncio.run(main(num_workers = 20))
except (KeyboardInterrupt, Exception) as e:
    # print("\nfinal exception:", e)
    print(traceback.format_exc())
    video_file.close()
    channel_file.close()