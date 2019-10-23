import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from tensorboardX import SummaryWriter
from itertools import count
import torch.nn.functional as F
import vizdoom as vzd
from torchvision import models
import cv2
import time
from PIL import Image
import torchvision


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # self.resnet = models.resnet18(pretrained=False)
        # # self.resnet = models.resnet50(pretrained=False)
        # self.relu = nn.ReLU()
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 64)
        #
        # self.fc_value = nn.Linear(64, 256)
        # self.fc_adv = nn.Linear(64, 256)
        #
        # self.value = nn.Linear(256, 1)
        # self.adv = nn.Linear(256, 3)

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 6, stride=2, padding=2)  # 64 * 64 * 3 -> 32 * 32 * 64

        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 32 * 32 * 64 -> 32 * 32 * 64
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 32 * 32 * 64 -> 32 * 32 * 64

        self.conv3 = nn.Conv2d(64, 64, 6, stride=2, padding=2)  # 32 * 32 * 64 -> 16 * 16 * 64

        self.conv4_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 16 * 16 * 64 -> 16 * 16 * 64
        self.conv4_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 6, stride=2, padding=2)  # 16 * 16 * 64 -> 8 * 8 * 64

        self.fc = nn.Linear(8 * 8 * 64, 1024)
        self.value = nn.Linear(1024, 1)
        self.adv = nn.Linear(1024, 3)


    def forward(self, x):
        # x = self.resnet(x)
        # value = self.relu(self.fc_value(x))
        # adv = self.relu(self.fc_adv(x))
        #
        # value = self.value(value)
        # adv = self.adv(adv)
        #
        # advAverage = torch.mean(adv, dim=1, keepdim=True)
        # Q = value + adv - advAverage

        x = self.relu(self.conv1(x))
        # print(x.shape)

        y = self.relu(self.conv2_1(x))
        y = self.conv2_2(y)
        # print(y.shape)
        x = self.relu(x + y)

        x = self.relu(self.conv3(x))

        y = self.relu(self.conv4_1(x))
        y = self.conv4_2(y)
        x = self.relu(x + y)

        x = self.relu(self.conv5(x))
        # print(x.shape)

        x = self.relu(self.fc(x.view(x.size(0), -1)))


        value = self.value(x)
        adv = self.adv(x)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


def preprocess(image_frame):  # transform w * h * d to d * w * h
    image_frame = cv2.resize(image_frame, (64, 64))
    trans = torchvision.transforms.ToTensor()
    image_frame = trans(image_frame)
    # print(image_frame)
    return image_frame.numpy()


onlineQNetwork = QNetwork().to(device)
targetQNetwork = QNetwork().to(device)
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-5)

GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 16

UPDATE_STEPS = 4

memory_replay = Memory(REPLAY_MEMORY)


epsilon = INITIAL_EPSILON
# epsilon = 0
learn_steps = 0
writer = SummaryWriter('doom_logs')
begin_learn = False

episode_reward = 0
previous_episode_reward = episode_reward

env = vzd.DoomGame()

# Sets path to additional resources wad file which is basically your scenario wad.
# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
env.set_doom_scenario_path("basic.wad")

# Sets map to start (scenario .wad files can contain many maps).
env.set_doom_map("map01")

# Sets resolution. Default is 320X240
env.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
env.set_screen_format(vzd.ScreenFormat.RGB24)

# Enables depth buffer.
# env.set_depth_buffer_enabled(True)

# Enables labeling of in env objects labeling.
# env.set_labels_buffer_enabled(True)

# Enables buffer with top down map of the current episode/level.
env.set_automap_buffer_enabled(True)

# Enables information about all objects present in the current episode/level.
env.set_objects_info_enabled(True)

# Enables information about all sectors (map layout).
env.set_sectors_info_enabled(True)

# Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
env.set_render_hud(False)
env.set_render_minimal_hud(False)  # If hud is enabled
env.set_render_crosshair(False)
env.set_render_weapon(True)
env.set_render_decals(False)  # Bullet holes and blood on the walls
env.set_render_particles(False)
env.set_render_effects_sprites(False)  # Smoke and blood
env.set_render_messages(False)  # In-env   messages
env.set_render_corpses(False)
env.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

# Adds buttons that will be allowed.
env.add_available_button(vzd.Button.MOVE_LEFT)
env.add_available_button(vzd.Button.MOVE_RIGHT)
env.add_available_button(vzd.Button.ATTACK)

# Adds env variables that will be included in state.
env.add_available_game_variable(vzd.GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
env.set_episode_timeout(200)

# Makes episodes start after 10 tics (~after raising the weapon)
env.set_episode_start_time(10)

# Makes the window appear (turned on by default)
env.set_window_visible(False)

# Turns on the sound. (turned off by default)
env.set_sound_enabled(False)

# Sets the livin reward (for each move) to -1
env.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
env.set_mode(vzd.Mode.PLAYER)

# Enables engine output to console.
#env.set_console_enabled(True)

# Initialize the env. Further configuration won't take any effect from now on.
env.init()

# move-left move-right attack
actions = [[True, False, False], [False, True, False], [False, False, True]]


epsilon = 0

# onlineQNetwork.load_state_dict(torch.load('doom-policy.para'))


def print_action(action, reward):
    if action == 0:
        print('move left, reward is : ', reward)
    elif action == 1:
        print('move right, reward is : ', reward)
    else:
        print('attack, reward is : ', reward)


for epoch in range(50000):
    env.new_episode()
    episode_reward = 0

    while not env.is_episode_finished():
        state = env.get_state()
        frame = preprocess(state.screen_buffer)
        p = random.random()
        if p < epsilon:
            action = random.randint(0, 2)
        else:
            tensor_frame = torch.FloatTensor(frame).unsqueeze(0).to(device)
            action = onlineQNetwork.select_action(tensor_frame)

        reward = env.make_action(actions[action])
        episode_reward += reward

        # print_action(action, reward)

        done = env.is_episode_finished()

        if done is True:
            next_frame = np.zeros((3, 64, 64))
        else:
            next_state = env.get_state()

            next_frame = preprocess(next_state.screen_buffer)

        memory_replay.add((frame, next_frame, action, reward, done))
        if memory_replay.size() > 1280:
            if begin_learn is False:
                print('learn begin!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)
            # print(np.asarray(batch_state).shape)
            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                onlineQ_next = onlineQNetwork(batch_next_state)
                targetQ_next = targetQNetwork(batch_next_state)
                online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

            loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=learn_steps)

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), 'doom-policy.para')
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))





