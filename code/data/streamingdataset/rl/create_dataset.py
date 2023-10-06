from envlogger import reader
import os
import logging
from streaming import MDSWriter
import time
from datetime import date, datetime
import numpy as np
import argparse


current_date = date.today()
current_datetime = datetime.now()

LEVEL_PATH = 'envs.txt'

# A dictionary of input fields to an Encoder/Decoder type
columns = {
    'obs_direction': 'ndarray',
    'obs_mission': 'ndarray',
    'obs_image': 'ndarray',
    'action': 'ndarray',
    'reward': 'ndarray',
    'discount': 'ndarray',
    'episode_id': 'ndarray'
}
# Compression algorithm name
compression = 'zstd'

# Hash algorithm name
hashes = 'sha1', 'xxh64'

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename=f'logs/output_{current_date}_{current_datetime}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(args):
    with open(LEVEL_PATH, 'r') as level_file:

        for level in level_file:
            logging.info('Level: %s', level.strip())

            for seed in range(args.total_random_seeds):
                logging.info('Seed: %d', seed)

                DATA_PATH = f'{args.data_path}/{args.agent}/{level.strip()}/{seed}'
                SAVE_PATH = f'{args.save_path}/{args.agent}/{level.strip()}/{seed}'
                os.makedirs(SAVE_PATH, exist_ok=True)

                episodes = []

                with reader.Reader(data_directory=DATA_PATH) as r:

                    total_episodes = 0
                    for episode_id, episode in enumerate(r.episodes):

                        obs_directions = []
                        obs_missions = []
                        obs_images = []
                        actions = []
                        rewards = []
                        discounts = []
                        episode_ids = []

                        for step in episode:
                            
                            direction = step.timestep.observation['direction']
                            mission = step.timestep.observation['mission']
                            mission = np.frombuffer(mission.encode('utf-8'), dtype=np.uint8)
                            image = step.timestep.observation['image']  # ndarray is not json serializable

                            reward = float(step.timestep.reward) if step.timestep.reward is not None else 0.0  # NOTE: Converting None to 0.0
                            action = int(step.action) if step.action is not None else -1  # NOTE: Adduming that -1 is not used as an action.
                            discount = float(step.timestep.discount) if step.timestep.discount is not None else 1.0  # NOTE: Converting None to 1.0
                            
                            obs_directions.append(direction)
                            obs_missions.append(mission)
                            obs_images.append(image)
                            actions.append(action)
                            rewards.append(reward)
                            discounts.append(discount)
                            episode_ids.append(episode_id)

                        multimodal_data_length = 5 * len(obs_directions)  # NOTE : 5 is the number of modalities, i.e., obs_direction, obs_mission, obs_image, action, reward

                        obs_directions_position_ids = np.arange(0, multimodal_data_length, 5)
                        obs_missions_position_ids = np.arange(1, multimodal_data_length + 1, 5)
                        obs_images_position_ids = np.arange(2, multimodal_data_length + 2, 5)
                        actions_position_ids = np.arange(3, multimodal_data_length + 3, 5)
                        rewards_position_ids = np.arange(4, multimodal_data_length + 4, 5)

                        multimodal_position_ids = np.concatenate((obs_directions_position_ids, obs_missions_position_ids, obs_images_position_ids, actions_position_ids, rewards_position_ids), axis=0)

                        episode_data = {
                            'obs_direction': np.asarray(obs_directions, dtype=np.int32),
                            'obs_mission': np.asarray(obs_missions, dtype=np.uint8),
                            'obs_image': np.asarray(obs_images, dtype=np.float32),
                            'action': np.asarray(actions, dtype=np.int32),
                            'reward': np.asarray(rewards, dtype=np.float32),
                            'discount': np.asarray(discounts, dtype=np.float32),
                            'episode_id': np.asarray(episode_ids, dtype=np.int32),
                            'multimodal_position_ids': np.asarray(multimodal_position_ids, dtype=np.int32)
                        }

                        episodes.append(episode_data)

                        total_episodes += 1

                # Start time
                start_time = time.time()
                with MDSWriter(out=SAVE_PATH, columns=columns, compression=compression, hashes=hashes) as save_file:
                    for episode_data in episodes:
                        save_file.write(episode_data)

                # End time
                end_time = time.time()

                elapsed_time = end_time - start_time

                logging.info('Total Episode: %d', total_episodes)
                logging.info('Elapsed Time: %f', elapsed_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='botagent', help='botagent or thoughtcloning')
    parser.add_argument('--data_path', type=str, default='/root/projects/rl-nlp/data-conversion/rlds')
    parser.add_argument('--save_path', type=str, default='/root/projects/videorl/code/data/streamingdataset/rl/mds')
    parser.add_argument('--total_random_seeds', type=int, default=10)
    args = parser.parse_args()
    main(args)