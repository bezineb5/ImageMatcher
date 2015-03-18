

# In-memory cache
ref_database = []


def clear():
    ref_database = []


def add(ref_image):
    ref_database.append(ref_image)


def get(index):
    return ref_database[index]
