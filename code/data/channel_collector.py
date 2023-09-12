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

channel_file = open('channels.tsv', 'a', encoding = "utf-8")
channel_writer = csv_writer(channel_file, delimiter = '\t')

# load all channels
channels = list()
with open('channels.txt', 'r') as f:
    for line in f:
        channels.append(line.strip())

# continue from previous run
with open('channels.tsv', 'r') as f:
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
        # remove channels already read (but reprocess last 100 channels)
        count = max(1, count - 100)
        channels = channels[:-count]


video_file = open('videos.tsv', 'a', encoding = "utf-8")
video_writer = csv_writer(video_file, delimiter = '\t')
with open('videos.tsv', 'r') as f:
    if f.readline() == '':
        video_writer.writerow(
            ['id', 'title', 'date', 'length', 'views']
        )



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
            # find the channel data within the html
            html = await response.text()
            soup = bs4(html, 'html.parser')
            data_str = 'var ytInitialData = '
            channel_data = json.loads(
                soup(text = re.compile(data_str))[0].strip(data_str).strip(';')
            )
            async with write_lock:
                channel_writer.writerow([channel_link] + parse_channel_info(channel_data))
                video_rows, token = parse_videos(channel_data)
                video_writer.writerows(video_rows)

        # continue "scrolling" through channel's videos
        while token is not None:
            data = copy.deepcopy(PAYLOAD)
            data['continuation'] = token
            async with session.post(
                BROWSE_ENDPOINT, headers = {'User-Agent': USER_AGENT},
                json = data, timeout = 5
            ) as response:
                video_data = await response.json()
                video_rows, token = parse_videos(video_data, is_continuation = True)
                async with write_lock:
                    video_writer.writerows(video_rows)


try:
    asyncio.run(collect_videos('/@realHeff'))
except Exception:
    print(traceback.format_exc())
    video_file.close()
    channel_file.close()