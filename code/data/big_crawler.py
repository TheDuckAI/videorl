import asyncio
import socket
import copy
import json
import os

from aiohttp import ClientSession, TCPConnector, DummyCookieJar

from youtube_helpers import BASE, ENDPOINT, PAYLOAD, USER_AGENT, parse_response



# windows specific fix
if os.name == 'nt': 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# locks for concurrency
processing_lock = asyncio.Lock()
print_lock = asyncio.Lock()

# data files
video_file = open('videos.txt', 'a')
channel_file = open('channels.txt', 'a')
playlist_file = open('playlists.jsonl', 'a', encoding = 'utf-8')

# channels seen so far (continue from previous runs)
channels = set()
with open('channels.txt', 'r') as f:
    for line in f:
        channels.add(line.strip())


async def get_recommendations(
        video_id, session, unexplored_videos,
        channel_set = channels, lock = processing_lock,
        video_file = video_file, channel_file = channel_file,
        playlist_file = playlist_file
    ):
    data = copy.deepcopy(PAYLOAD)
    data['videoId'] = video_id
    async with session.post(ENDPOINT, headers = {'User-Agent': USER_AGENT},
                            json = data, timeout = 5) as response:
        if response.ok is False:
            return response.ok
        
        recommendations = parse_response(await response.json())
        for recommendation in recommendations:
            async with lock:
                video_file.write(recommendation['id'] + '\n')

                if recommendation['isPlaylist']:
                    playlist_file.write(json.dumps(recommendation) + '\n')
                
                if (recommendation['channel']['link'] is not None and
                    recommendation['channel']['link'] not in channel_set):

                    channel_set.add(recommendation['channel']['link'])
                    channel_file.write(recommendation['channel']['link'] + '\n')

                    if ('shorts' not in recommendation['link'] and
                        recommendation['isPlaylist'] is not True):
                        unexplored_videos.append(recommendation['id'])


async def worker(unexplored_videos, num_reqs, channels_set = channels):
    async with print_lock:
        print('worker started')

    while True:
        # use ipv6 (helps with blocks) and leave concurrency to parallel connections
        conn = TCPConnector(limit = 1, family = socket.AF_INET6, force_close = True)
        async with ClientSession(
            base_url = BASE, connector = conn, cookie_jar = DummyCookieJar()
        ) as session:
            for _ in range(num_reqs):
                if len(unexplored_videos) == 0:
                    async with print_lock:
                        print('no more videos to explore, stopping worker')
                    return

                video_id = unexplored_videos.pop()
                try:
                    ok_response = await get_recommendations(video_id, session, unexplored_videos)
                    if ok_response is False:
                        async with print_lock:
                            print("bad response, stopping worker (try restarting)")
                        return
                except Exception as e:
                    async with print_lock:
                        print(e, video_id)
        async with print_lock:
            print(
                'finished connection, number of channels:',
                len(channels_set), end = "\t\t\t\r"
            )
            

async def main(num_workers, num_reqs):
    # read last num_workers * 1000 in to start the crawler back up
    initial_videos = os.popen(
        'tail -n ' + str(num_workers * 1000) + ' videos.txt'
    ).read().split('\n')[:-1]

    # if videos.txt doesn't have enough videos (cold start), fill it with some recommendations
    if len(initial_videos) == 0:
        assert len(channels) == 0, \
            'channels.txt should be empty for cold start, delete channels.txt and try again'

        # start with an old and popular video
        initial_videos = ['dQw4w9WgXcQ']
        async with ClientSession(base_url = BASE) as session:
            # collect num_workers * num_reqs videos (just a heuristic)
            while len(initial_videos) < num_workers * num_reqs:
                video_id = initial_videos.pop()
                await get_recommendations(video_id, session, initial_videos)
                print(
                    f'collecting initial videos: {len(initial_videos)}/{num_workers * num_reqs}',
                    end = '\t\t\t\r'
                )
        print('\nfinished collecting initial videos, starting asyncronous workers')
    else:
        print('loaded previous videos.txt and channels.txt, starting asyncronous workers')

    # split unexplored videos equally among workers
    await asyncio.gather(*[
        worker(copy.deepcopy(initial_videos[i::num_workers]), num_reqs)
        for i in range(num_workers)
    ])


try:
    # launch 20 concurrent workers that each make 20 requests before restarting connection
    # this would be the high end of normal individual youtube traffic
    asyncio.run(main(num_workers = 20, num_reqs = 20))
except (KeyboardInterrupt, Exception) as e:
    print("\nfinal exception:", e)

    # make sure to exit cleanly
    video_file.flush()
    channel_file.flush()
    playlist_file.flush()
    video_file.close()
    channel_file.close()
    playlist_file.close()