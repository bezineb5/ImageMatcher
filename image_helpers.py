from PIL import Image

def generate_thumbnail(input_file, output_file, size):
    input_file.seek(0)
    im = Image.open(input_file)
    im.thumbnail(size)
    im.save(output_file, "JPEG", quality=50)
