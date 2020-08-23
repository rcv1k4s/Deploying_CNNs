import requests

def send_image(ip,port,image_name):
    url = 'http://{}:{}/im_process'.format(ip, port)
    my_img = {'image': open(image_name, 'rb')}
    ret = requests.post(url, files=my_img)
    print(ret.content)

if __name__ == '__main__':
    ip = '0.0.0.0'
    port = 7718
    image_name = 'test_input.png'
    send_image(ip,port,image_name)
