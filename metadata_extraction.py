import exifread
from werkzeug.utils import secure_filename

def compute_single_ratio(ratio):
	return ratio.num / ratio.den

def compute_coordinate_rational(ratio_list):
	degrees = compute_single_ratio(ratio_list[0])
	minutes = compute_single_ratio(ratio_list[1])
	seconds = compute_single_ratio(ratio_list[2])

	return degrees + (minutes + seconds / 60.0) / 60.0

def get_coordinate(field_ratio, field_sign, tags):
	if field_ratio in tags and field_sign in tags:
		sign_desc = tags[field_sign].values
		ratio_list = tags[field_ratio].values
		
		value = compute_coordinate_rational(ratio_list)
		if sign_desc == 'W' or sign_desc == 'S':
			value = -value

		return value
	else:
		return None

def extract_metadata(file):
    # load the EXIF data
    file.seek(0)
    tags = exifread.process_file(file, details=False)
    print tags

    # title
    name = None
    if "Image ImageDescription" in tags:
    	name = tags['Image ImageDescription'].values
    else:
    	name = secure_filename(file.filename)

    # GPS coordinates
    longitude = get_coordinate('GPS GPSLongitude', 'GPS GPSLongitudeRef', tags)
    latitude = get_coordinate('GPS GPSLatitude', 'GPS GPSLatitudeRef', tags)
    altitude = compute_single_ratio(tags['GPS GPSAltitude'].values[0]) if 'GPS GPSAltitude' in tags else None
    metadata = { "name": name, "latitude":latitude, "longitude": longitude, "altitude": altitude }

    return metadata
