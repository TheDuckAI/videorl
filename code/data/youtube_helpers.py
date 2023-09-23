from datetime import datetime
from dateutil.relativedelta import relativedelta



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
    

def parse_date(date):
    val, unit = date.split()[:2]
    if not unit.endswith('s'):
        unit = unit + 's'
    past_time = datetime.now() - relativedelta(**{unit:int(val)})
    return past_time.strftime("%Y-%m-%d")


def get_tab(response, name):
    tabs = getValue(response, ['contents', 'twoColumnBrowseResultsRenderer', 'tabs'])
    for tab in tabs:
        if 'tabRenderer' in tab and getValue(tab, ['tabRenderer', 'title']) == name:
            return getValue(tab, ['tabRenderer', 'content'])


def get_continuation(response):
    continuation = getValue(
            response, ['onResponseReceivedActions', 0, 'appendContinuationItemsAction', 'continuationItems']
        )

    # continuation returned empty (i.e. no more videos to collect)
    if 'responseContext' in response and continuation is None:
        return None
    
    return continuation


################################### TAB PARSERS ####################################
def parse_about(channel_link, response = None, tab = None, **kwargs):
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
    badges = getValue(response, ['header', 'c4TabbedHeaderRenderer', 'badges'])
    if badges is not None:
        for badge in badges:
            if getValue(badge, ['metadataBadgeRenderer', 'tooltip']) == 'Verified':
                verified = True

    full_meta = getValue(tab, ['sectionListRenderer', 'contents', 0, 'itemSectionRenderer', 'contents', 0, 'channelAboutFullMetadataRenderer'])
    view_count = getValue(full_meta, ['viewCountText', 'simpleText'])
    if view_count is not None:
        view_count = text_to_num(view_count.split(' ')[0].replace(',', ''))

    join_date = getValue(full_meta, ['joinedDateText', 'runs', 1, 'text'])
    if join_date is not None:
        join_date = datetime.strptime(join_date, '%b %d, %Y').strftime("%Y-%m-%d")

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

    # use 2d array to match the format of other parsers)
    return [[channel_link, id, title, subscribers, view_count, num_vids_shorts, join_date, country, monetization, verified,
            isFamilySafe, description, tags, fullsize_avatar, fullsize_banner, parsed_links]], None



def parse_videos(channel_link, tab = None, continuation = None, **kwargs):
    video_infos = None
    if continuation is None:
        # initial get request, ensure there are videos to begin with
        video_infos = getValue(
            tab, ['richGridRenderer', 'contents']
        )

        # video tab doesn't contain videos
        if video_infos is None:
            return None, None
    else:
        video_infos = continuation

    assert video_infos is not None, "Unable to find videos in response"

    video_rows = []
    token = None
    for info in video_infos:
        if "richItemRenderer" in info:
            video_data = getValue(info, ['richItemRenderer', 'content', 'videoRenderer'])

            if 'publishedTimeText' not in video_data:
                # unpublished video for the future
                continue

            id = video_data["videoId"]
            title = getValue(video_data, ['title', 'runs', 0, 'text'])
            publish = parse_date(video_data['publishedTimeText']['simpleText'].replace('Streamed ', ''))
            length = getValue(video_data, ['lengthText', 'simpleText'])
            views = getValue(video_data, ['viewCountText', 'simpleText'])
            if views is not None:
                views = int(views.split(' ')[0].replace(',', '').replace('No', '0'))
            description_snippet = getValue(video_data, ['descriptionSnippet', 'runs', 0, 'text'])

            thumbnails = getValue(video_data, ['thumbnail', 'thumbnails'])
            moving_thumbnails = getValue(video_data, ['richThumbnail', 'movingThumbnailRenderer', 'movingThumbnailDetails', 'thumbnails'])


            video_rows.append([channel_link, id, title, publish, length, views, description_snippet, moving_thumbnails, thumbnails])
        if 'continuationItemRenderer' in info:
            token = getValue(info, ['continuationItemRenderer', 'continuationEndpoint', 'continuationCommand', 'token'])
    return video_rows, token


def parse_shorts(channel_link, tab = None, continuation = None, **kwargs):
    if continuation is None:
        # initial request, ensure there are shorts to begin with
        shorts_info = getValue(
            tab, ['richGridRenderer', 'contents']
        )

        # shorts tab empty
        if shorts_info is None:
            return None, None
    else:
        shorts_info = continuation

    assert shorts_info is not None, "Unable to find shorts in response"
    
    shorts = []
    token = None
    for renderer in shorts_info:
        if "richItemRenderer" in renderer:
            short_content = getValue(renderer, ["richItemRenderer", "content", "reelItemRenderer"])
            id = getValue(short_content, ["videoId"])
            headline = getValue(short_content, ["headline", "simpleText"])

            thumbnail = getValue(short_content, ["thumbnail", "thumbnails", 0])
            width = None
            height = None
            if thumbnail is not None and 'url' in thumbnail:
                width = getValue(thumbnail, ["width"])
                height = getValue(thumbnail, ["height"])
                thumbnail = getValue(thumbnail, ["url"])

            viewCountText = getValue(short_content, ["viewCountText", "simpleText"])
            if viewCountText is not None:
                viewCountText = text_to_num(viewCountText.split(' ')[0].replace(',', '').replace('No', '0'))
            
            length = getValue(short_content, ["accessibility", "accessibilityData", "label"])
            if length is not None:
                length = length.split(' - ')[1]
            
            shorts.append([channel_link, id, headline, viewCountText, length, thumbnail, width, height])
        if 'continuationItemRenderer' in renderer:
            token = getValue(renderer, ['continuationItemRenderer', 'continuationEndpoint', 'continuationCommand', 'token'])
    return shorts, token


def parse_live(channel_link, tab = None, continuation = None, **kwargs):
    return parse_videos(channel_link, tab = tab, continuation = continuation)


def parse_playlists(channel_link, tab = None, continuation = None, **kwargs):
    if continuation is None:
        # initial request, ensure there are playlists to begin with
        playlist_infos = getValue(tab,
            ['sectionListRenderer', 'contents', 0, 'itemSectionRenderer', 'contents', 0, 'gridRenderer', 'items']
        )

        # playlist tab empty
        if playlist_infos is None:
            return None, None
    else:
        playlist_infos = continuation

    assert playlist_infos is not None, "Unable to find playlists in response"

    playlists = []
    token = None
    for info in playlist_infos:
        if 'gridPlaylistRenderer' in info:
            info = info['gridPlaylistRenderer']
            id = getValue(info, ['playlistId'])
            title = getValue(info, ['title', 'runs', 0, 'text'])
            num_videos = getValue(info, ['videoCountShortText', 'simpleText'])

            playlists.append([channel_link, id, title, num_videos])
        if 'continuationItemRenderer' in info:
            token = getValue(info, ['continuationItemRenderer', 'continuationEndpoint', 'continuationCommand', 'token'])

    return playlists, token


def parse_featured_channels(channel_link, tab = None, continuation = None, **kwargs):
    if continuation is None:
        # initial request, ensure there are featured channels to begin with
        channel_infos = getValue(tab,
            ['sectionListRenderer', 'contents', 0, 'itemSectionRenderer', 'contents', 0, 'gridRenderer', 'items']
        )

        # featured channels tab empty
        if channel_infos is None:
            return None, None
    else:
        channel_infos = continuation

    assert channel_infos is not None, "Unable to find featured channels in response"

    channels = []
    token = None
    for info in channel_infos:
        if 'gridChannelRenderer' in info:
            id = getValue(info, ['channelId'])
            url = getValue(info, ['navigationEndpoint', 'commandMetadata', 'webCommandMetadata', 'url'])
            name = getValue(info, ['title', 'simpleText'])
            subscribers = getValue(info, ['subscriberCountText', 'simpleText'])
            num_shorts_vids = getValue(info, ['videoCountText', 'runs', 0, 'text'])

            channels.append([channel_link, id, url, name, subscribers, num_shorts_vids])
        if 'continuationItemRenderer' in info:
            token = getValue(info, ['continuationItemRenderer', 'continuationEndpoint', 'continuationCommand', 'token'])
    return channels, token



# organize the parsers
tab_helpers = [
    {
        'name': 'About', 'filename': 'channels', 'param': 'EgVhYm91dPIGBAoCEgA%3D', 'parse_func': parse_about,
        'features': [
            'link', 'id', 'title', 'subscribers', 'view_count', 'num_vids_shorts', 'join_date', 'country', 'monetization', 'verified',
            'isFamilySafe', 'description', 'tags', 'fullsize_avatar', 'fullsize_banner', 'parsed_links'
        ]
    },
    {
        'name': 'Videos', 'filename': 'videos', 'param': 'EgZ2aWRlb3PyBgQKAjoA', 'parse_func': parse_videos,
        'features': ['link', 'id', 'title', 'approx_date', 'length', 'views', 'description_snippet', 'moving_thumbnails', 'thumbnails']
    },
    {
        'name': 'Shorts', 'filename': 'shorts', 'param': 'EgZzaG9ydHPyBgUKA5oBAA%3D%3D', 'parse_func': parse_shorts,
        'features': ['link', 'id', 'headline', 'viewCountText', 'length', 'thumbnail', 'width', 'height']
    },
    {
        'name': 'Live', 'filename': 'livestreams', 'param': 'EgdzdHJlYW1z8gYECgJ6AA%3D%3D', 'parse_func': parse_live,
        'features': ['link', 'id', 'title', 'approx_date', 'length', 'views', 'description_snippet', 'moving_thumbnails', 'thumbnails']
    },
    {
        'name': 'Playlists', 'filename': 'playlists', 'param': 'EglwbGF5bGlzdHPyBgQKAkIA', 'parse_func': parse_playlists,
        'features': ['link', 'id', 'title', 'num_videos']
    },
    {
        'name': 'Channels', 'filename': 'featured_channels', 'param': 'EghjaGFubmVsc_IGBAoCUgA%3D', 'parse_func': parse_featured_channels,
        'features': ['link', 'id', 'url', 'name', 'subscribers', 'num_shorts_vids']
    },
]
