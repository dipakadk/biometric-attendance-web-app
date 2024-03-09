import cv2
import face_recognition
import pickle
import os

# importing images of student
folder_user_path = 'Images'
user_path = os.listdir(folder_user_path)
user_img_list = []
user_ids = []
for i in user_path:
    user_img_list.append(cv2.imread(os.path.join(folder_user_path, i)))
    # print(os.path.splitext(i)[0])
    user_ids.append(os.path.splitext(i)[0])


# print(len(student_list))
# print(student_ids)


def encoding_images(images):
    encode_image_list = []
    for img in user_img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_image_list.append(encode)
    return encode_image_list


known_encode_list = encoding_images(user_img_list)
known_encode_list_with_ids = [known_encode_list, user_ids]

print(known_encode_list_with_ids)

file = open('encoding_file.p', 'wb')
pickle.dump(known_encode_list_with_ids, file)
file.close()
print("File saved successfully!!!!")
