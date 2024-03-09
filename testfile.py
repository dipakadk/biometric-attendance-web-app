def convert_to_binary(filename):
    with open(filename, "rb") as f:
        binary_data = f.read()
    return binary_data

my_image_path= 'Images/Dipak Adhikari.jpg'
print(convert_to_binary(my_image_path))
