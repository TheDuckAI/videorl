import dateparser

# constants used for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0"
BASE = 'https://www.youtube.com'
ENDPOINT = '/youtubei/v1/next?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8&prettyPrint=false'
BROWSE_ENDPOINT = '/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8&prettyPrint=false'
PAYLOAD = {
    "context": {
        "client": {
            "clientName": "WEB",
            "clientVersion": "2.20230921.01.00",
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


denominations = {
        'K': 1000,
        'M': 1000000,
        'B': 1000000000
    }
def text_to_num(text, demoninations = denominations):
    if text[-1].upper() in denominations:
        # separate out the K, M, or B
        num, magnitude = text[:-1], text[-1]
        return int(float(num) * denominations[magnitude])
    else:
        return int(text)
    

tab_params = {
    'Videos'    :   'EgZ2aWRlb3PyBgQKAjoA',
    'Shorts'    :   'EgZzaG9ydHPyBgUKA5oBAA%3D%3D',
    'Live'      :   'EgdzdHJlYW1z8gYECgJ6AA%3D%3D',
    'Playlists' :   'EglwbGF5bGlzdHPyBgQKAkIA',
    'Channels'  :   'EghjaGFubmVsc_IGBAoCUgA%3D',
    'About'     :   'EgVhYm91dPIGBAoCEgA%3D'
}


def get_tab(response, name):
    tabs = getValue(response, ['contents', 'twoColumnBrowseResultsRenderer', 'tabs'])
    for tab in tabs:
        if 'tabRenderer' in tab and getValue(tab, ['tabRenderer', 'title']) == name:
            return getValue(tab, ['tabRenderer', 'content'])


def parse_channel_info(response, channel_link):
    id = getValue(response, ["metadata", "channelMetadataRenderer", "externalId"])

    title = getValue(response, ["metadata", "channelMetadataRenderer", "title"])

    subscribers = getValue(
        response,
        ["header", "c4TabbedHeaderRenderer", "subscriberCountText", "simpleText"]
    )
    if subscribers is not None:
        subscribers = text_to_num(subscribers.split(' ')[0])

    num_vids_shorts = getValue(response, ["header", "c4TabbedHeaderRenderer", "videosCountText", "runs", 0, "text"])
    if num_vids_shorts is not None:
        num_vids_shorts = text_to_num(num_vids_shorts.split(' ')[0].replace(',', '').replace('No', '0'))

    description = getValue(response, ["metadata", "channelMetadataRenderer", "description"])
    isFamilySafe = getValue(response, ["metadata", "channelMetadataRenderer", "isFamilySafe"])

    monetization = None
    for param in getValue(response, ['responseContext', 'serviceTrackingParams']):
        if param.get('service') == 'GFEEDBACK':
            for feedback in param['params']:
                key = feedback.get('key')
                if key == 'is_monetization_enabled':
                    monetization = (feedback.get('value') == 'true')

    verified = False
    for badge in getValue(response, ['header', 'c4TabbedHeaderRenderer', 'badges']):
        if getValue(badge, ['metadataBadgeRenderer', 'tooltip']) == 'Verified':
            verified = True

    about = get_tab(response, 'About')
    full_meta = getValue(about, ['sectionListRenderer', 'contents', 0, 'itemSectionRenderer', 'contents', 0, 'channelAboutFullMetadataRenderer'])
    view_count = getValue(full_meta, ['viewCountText', 'simpleText'])
    if view_count is not None:
        view_count = text_to_num(view_count.split(' ')[0].replace(',', ''))

    join_date = getValue(full_meta, ['joinedDateText', 'runs', 1, 'text'])
    if join_date is not None:
        join_date = dateparser.parse(join_date).strftime("%Y-%m-%d")

    country = getValue(full_meta, ['country', 'simpleText'])

    tags = getValue(response, ["microformat", "microformatDataRenderer", "tags"])

    fullsize_avatar = getValue(full_meta, ['avatar', 'thumbnails', 0, 'url'])
    if fullsize_avatar is not None:
        fullsize_avatar = fullsize_avatar.split('=')[0]

    fullsize_banner = getValue(response, ["header", "c4TabbedHeaderRenderer", "banner", 'thumbnails', 0, 'url'])
    if fullsize_banner is not None:
        fullsize_banner = fullsize_banner.split('=')[0]

    links = full_meta.get('links')
    parsed_links = None
    if links is not None:
        parsed_links = []
        for link in links:
            link_title = getValue(link, ['channelExternalLinkViewModel', 'title', 'content'])
            link_url = getValue(link, ['channelExternalLinkViewModel', 'link', 'content'])
            parsed_links.append({'title': link_title, 'link': link_url})
    parsed_links


    return [channel_link, id, title, subscribers, view_count, num_vids_shorts, join_date, country, monetization, verified,
            isFamilySafe, description, tags, fullsize_avatar, fullsize_banner, parsed_links]



# THIS CODE IS QUITE FRAGILE TO YOUTUBE CHANGING ITS UI, MAKE SURE THERE AREN'T TOO MANY NANs in OUTPUTS
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
        video_tab = None
        for tab in tabs:
            if 'tabRenderer' in tab and getValue(tab, ['tabRenderer', 'title']) == "Videos":
                video_tab = tab
        
        # no videos in channel
        if video_tab is None:
            return None, None
                
        video_infos = getValue(
            video_tab, ['tabRenderer', 'content', 'richGridRenderer', 'contents']
        )

        # video tab doesn't contain videos
        if video_infos is None:
            return None, None

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
            publish = dateparser.parse(video_data['publishedTimeText']['simpleText']).strftime("%Y-%m-%d")
            length = getValue(video_data, ['lengthText', 'simpleText'])
            views = getValue(video_data, ['viewCountText', 'simpleText'])
            if views is not None:
                views = int(views.split(' ')[0].replace(',', '').replace('No', '0'))
            description_snippet = getValue(video_data, ['descriptionSnippet', 'runs', 0, 'text'])

            video_rows.append([channel_link, id, title, publish, length, views, description_snippet])
    return video_rows, token