import time
import json
import pickle
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='Streams a file to a Spark Streaming Context')
parser.add_argument('--folder', '-f', help='Data folder', default="./Animals-10", required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream',required=False, type=bool, default=False)
parser.add_argument('--split','-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep','-t', help="streaming interval", required=False, type=int, default=3)

TCP_IP = "localhost"
TCP_PORT = 1412

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.labels = []

    def data_generator(self, data_file: str, batch_size: int):
        batch = []
        with open(data_file, "rb") as batch_file:
            batch_data = pickle.load(batch_file, encoding='bytes')
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])

        data = np.vstack(self.data)
        self.data = list(map(np.ndarray.tolist, data))
        size_per_batch = (len(self.data)//batch_size)*batch_size
        for ix in range(0, size_per_batch, batch_size):
            image = self.data[ix:ix+batch_size]
            label = self.labels[ix:ix+batch_size]
            batch.append([image, label])
        
        self.data = self.data[ix+batch_size:]
        self.labels = self.labels[ix+batch_size:]
        
        return batch

    def sendAnimalBatchFileToSpark(self, tcp_connection, input_batch_file, batch_size, split="train"):
        if split == "train":
            total_batch = 50_000 / batch_size + 1
        else:
            total_batch = 10_000 / batch_size + 1

        pbar = tqdm(total_batch)
        data_received = 0
        for file in input_batch_file:
            batches = self.data_generator(file, batch_size)
            for batch in batches:
                images, labels = batch
                images = np.array(images)
                images = images.reshape(images.shape[0], -1)
                batch_size, feature_size = images.shape
                images = images.tolist()

                payload = dict()
                for batch_idx in range(batch_size):
                    payload[batch_idx] = dict()
                    for feature_idx in range(feature_size):
                        payload[batch_idx][f'feature-{feature_idx}'] = images[batch_idx][feature_idx]
                    payload[batch_idx]['label'] = labels[batch_idx]

                # convert the payload to string
                payload = (json.dumps(payload) + "\n").encode()
                try:
                    tcp_connection.send(payload)
                except BrokenPipeError:
                    print("Either batch size is too big for the dataset or the connection was closed")
                except Exception as error_message:
                    print(f"Exception thrown but was handled: {error_message}")

                data_received += 1
                pbar.update(n=1)
                pbar.set_description(f"it: {data_received} | received : {batch_size} images")
                time.sleep(sleep_time)

    def connectTCP(self):   
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")

        return connection, address

    def streamAnimalDataset(self, tcp_connection, folder, batch_size):
        ANIMAL_BATCHES = [
            os.path.join(folder, 'data_batch_1'),
            os.path.join(folder, 'data_batch_2'),
            os.path.join(folder, 'data_batch_3'),
            os.path.join(folder, 'data_batch_4'),
            os.path.join(folder, 'data_batch_5'),
            os.path.join(folder, 'test_batch'),
        ]
        ANIMAL_BATCHES = ANIMAL_BATCHES[:-1] if train_test_split=='train' else [ANIMAL_BATCHES[-1]]
        self.sendAnimalBatchFileToSpark(tcp_connection, ANIMAL_BATCHES, batch_size, train_test_split)

if __name__ == '__main__':
    args = parser.parse_args()

    data_folder = args.folder
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()
    
    if endless:
        while True:
            dataset.streamAnimalDataset(tcp_connection, data_folder, batch_size)
    else:
        dataset.streamAnimalDataset(tcp_connection, data_folder, batch_size)

    tcp_connection.close()
