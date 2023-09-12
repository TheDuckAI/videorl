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
        self.file.write(self.delimiter.join(row) + '\n')
    
    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def tsv_clean(dirty_str):
    return dirty_str.replace('\n', '').replace('\t', '')


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


def parse_channel_info(response):
    title =  getValue(response, ["metadata", "channelMetadataRenderer", "title"])
    description = tsv_clean(getValue(response, ["metadata", "channelMetadataRenderer", "description"]))
    subscribers = text_to_num(getValue(
        response,
        ["header", "c4TabbedHeaderRenderer", "subscriberCountText", "simpleText"]
    ).split(' ')[0])
    isFamilySafe = str(getValue(response, ["metadata", "channelMetadataRenderer", "isFamilySafe"]))
    tags = getValue(response, ["microformat", "microformatDataRenderer", "tags"])
    if tags is not None:
        tags = tsv_clean(', '.join(tags))
    return [title, description, subscribers, isFamilySafe, tags]


def parse_videos(response, is_continuation = False):
    video_infos = None
    if is_continuation:
        video_infos = getValue(
            response, ['onResponseReceivedActions', 0, 'appendContinuationItemsAction', 'continuationItems']
        )
    else:
        # initial get request, ensure there are videos to begin with
        tabs = getValue(response, ['contents', 'twoColumnBrowseResultsRenderer', 'tabs'])
        video_tab_index = None
        for i, tab in enumerate(tabs):
            if 'tabRenderer' in tab and getValue(tab, ['tabRenderer', 'title']) == "Videos":
                video_tab_index = i
        
        if video_tab_index is None:
            return None, None

        video_infos = getValue(
            response, ['contents', 'twoColumnBrowseResultsRenderer', 'tabs', video_tab_index, 'tabRenderer', 'content', 'richGridRenderer', 'contents']
        )

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
            title = tsv_clean(video_data['title']['runs'][0]['text'])
            # description = video_data['descriptionSnippet']['runs'][0]['text']
            publish = dateparser.parse(video_data['publishedTimeText']['simpleText']).strftime("%Y-%m-%d")
            length = video_data['lengthText']['simpleText']
            views = str(int(video_data['viewCountText']['simpleText'].split(' ')[0].replace(',', '').replace('No', '0')))
            video_rows.append([id, title, publish, length, views])
    return video_rows, token