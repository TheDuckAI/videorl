import dateparser

# constants used for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0"
BASE = 'https://www.youtube.com'
ENDPOINT = '/youtubei/v1/next?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8&prettyPrint=false'
BROWSE_ENDPOINT = '/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8&prettyPrint=false'
PAYLOAD = {
    "context": {
        "client": {
            "clientName": "WEB",
            "clientVersion": "2.20230831.09.00",
            "newVisitorCookie": True,
        },
        "user": {
            "lockedSafetyMode": False,
        }
    }
}


def getValue(source, path):
    value = source
    for key in path:
        if type(key) is str:
            if key in value.keys():
                value = value[key]
            else:
                value = None
                break
        elif type(key) is int:
            if len(value) != 0:
                value = value[key]
            else:
                value = None
                break
    return value


def parse_response(response):
    videorenderers = getValue(response, ["playerOverlays", "playerOverlayRenderer", "endScreen", "watchNextEndScreenRenderer", "results"])
    videos = []

    if videorenderers is None:
        return videos

    for video in videorenderers:
        if "endScreenVideoRenderer" in video.keys():
            video = video["endScreenVideoRenderer"]
            j = {
                "isPlaylist" : False,
                "id": getValue(video, ["videoId"]),
                "thumbnails": getValue(video, ["thumbnail", "thumbnails"]),
                "title": getValue(video, ["title", "simpleText"]),
                "channel": {
                    "name": getValue(video, ["shortBylineText", "runs", 0, "text"]),
                    "id": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "browseId"]),
                    "link": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "canonicalBaseUrl"]),
                },
                "duration": getValue(video, ["lengthText", "simpleText"]),
                "accessibility": {
                    "title": getValue(video, ["title", "accessibility", "accessibilityData", "label"]),
                    "duration": getValue(video, ["lengthText", "accessibility", "accessibilityData", "label"]),
                },
                "link": "https://www.youtube.com" + getValue(video, ["navigationEndpoint", "commandMetadata", "webCommandMetadata", "url"]),
                "isPlayable": getValue(video, ["isPlayable"]),
                "videoCount": 1,
            }
            videos.append(j)

        if "endScreenPlaylistRenderer" in video.keys():
            video = video["endScreenPlaylistRenderer"]
            j = {
                "isPlaylist" : True,
                "id": getValue(video, ["playlistId"]),
                "thumbnails": getValue(video, ["thumbnail", "thumbnails"]),
                "title": getValue(video, ["title", "simpleText"]),
                "channel": {
                    "name": getValue(video, ["shortBylineText", "runs", 0, "text"]),
                    "id": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "browseId"]),
                    "link": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "canonicalBaseUrl"]),
                },
                "duration": getValue(video, ["lengthText", "simpleText"]),
                "accessibility": {
                    "title": getValue(video, ["title", "accessibility", "accessibilityData", "label"]),
                    "duration": getValue(video, ["lengthText", "accessibility", "accessibilityData", "label"]),
                },
                "link": "https://www.youtube.com" + getValue(video, ["navigationEndpoint", "commandMetadata", "webCommandMetadata", "url"]),
                "isPlayable": getValue(video, ["isPlayable"]),
                "videoCount": getValue(video, ["videoCount"]),
            }
            videos.append(j)
    return videos


# basic csvwriter (re-writing since csvwriter can't handle append mode)
class csv_writer:
    def __init__(self, file, delimiter = '\t'):
        self.file = file
        self.delimiter = delimiter
    
    def writerow(self, row):
        assert type(row) is list, "attempting to write non-list row"
        for i in range(len(row)):
            if row[i] is None:
                row[i] = ''
        self.file.write(self.delimiter.join(row) +  "\r\n")
    
    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def tsv_clean(dirty_str):
    if dirty_str is None:
        return None
    return dirty_str.strip().replace('\n', '<newline>').replace('\t', '').replace('\r', '')


denominations = {
        'K': 1000,
        'M': 1000000,
        'B': 1000000000
    }
def text_to_num(text):
    if text[-1] in denominations:
        # separate out the K, M, or B
        num, magnitude = text[:-1], text[-1]
        return str(int(float(num) * denominations[magnitude]))
    else:
        return str(int(text))


def parse_channel_info(response, channel_link):
    id = getValue(response, ["metadata", "channelMetadataRenderer", "externalId"])
    
    title =  getValue(response, ["metadata", "channelMetadataRenderer", "title"])
    
    subscribers = getValue(
        response,
        ["header", "c4TabbedHeaderRenderer", "subscriberCountText", "simpleText"]
    )
    if subscribers is not None:
        subscribers = text_to_num(subscribers.split(' ')[0])
    
    num_vids_shorts = getValue(response, ["header", "c4TabbedHeaderRenderer", "videosCountText", "runs", 0, "text"])
    if num_vids_shorts is not None:
        num_vids_shorts = text_to_num(num_vids_shorts.split(' ')[0].replace(',', '').replace('No', '0'))
    
    description = tsv_clean(getValue(response, ["metadata", "channelMetadataRenderer", "description"]))
    
    isFamilySafe = getValue(response, ["metadata", "channelMetadataRenderer", "isFamilySafe"])
    if isFamilySafe is not None:
        isFamilySafe = str(isFamilySafe)
    
    keywords = getValue(response, ["metadata", "channelMetadataRenderer", "keywords"])
    if keywords is not None:
        keywords = tsv_clean(keywords)

    return [channel_link, id, title, subscribers, num_vids_shorts, description, isFamilySafe, keywords]


def parse_videos(response, channel_link, is_continuation = False):
    video_infos = None
    if is_continuation:
        video_infos = getValue(
            response, ['onResponseReceivedActions', 0, 'appendContinuationItemsAction', 'continuationItems']
        )

        # continuation returned empty (i.e. no more videos to collect)
        if 'responseContext' in response and video_infos is None:
            return None, None
    else:
        # initial get request, ensure there are videos to begin with
        tabs = getValue(response, ['contents', 'twoColumnBrowseResultsRenderer', 'tabs'])
        video_tab_index = None
        for i, tab in enumerate(tabs):
            if 'tabRenderer' in tab and getValue(tab, ['tabRenderer', 'title']) == "Videos":
                video_tab_index = i
        
        # no videos in channel
        if video_tab_index is None:
            return None, None

        video_infos = getValue(
            response, ['contents', 'twoColumnBrowseResultsRenderer', 'tabs', video_tab_index, 'tabRenderer', 'content', 'richGridRenderer', 'contents']
        )

    assert video_infos is not None, "Unable to find videos in response"

    video_rows = []
    token = None
    for info in video_infos:
        if 'continuationItemRenderer' in info:
            token = getValue(info, ['continuationItemRenderer', 'continuationEndpoint', 'continuationCommand', 'token'])
        else:
            video_data = getValue(info, ['richItemRenderer', 'content', 'videoRenderer'])

            if 'publishedTimeText' not in video_data:
                # unpublished video for the future
                continue

            id = video_data["videoId"]
            title = getValue(video_data, ['title', 'runs', 0, 'text'])

            if title is not None:
                title = tsv_clean(title)

            publish = dateparser.parse(video_data['publishedTimeText']['simpleText']).strftime("%Y-%m-%d")

            length = getValue(video_data, ['lengthText', 'simpleText'])

            views = getValue(video_data, ['viewCountText', 'simpleText'])
            if views is not None:
                views = str(int(views.split(' ')[0].replace(',', '').replace('No', '0')))

            description_snippet = getValue(video_data, ['descriptionSnippet', 'runs', 0, 'text'])

            video_rows.append([channel_link, id, title, publish, length, views, description_snippet])
    return video_rows, token