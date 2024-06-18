import numpy as np
import time
import os
#import gc

#import matplotlib.pyplot as plt
import copy
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix, csc_matrix
import tkinter as tk
from tkinter import messagebox
from environment import Env
from environment import Agent

seed = 1
np.random.seed(seed)

np.set_printoptions(precision=3)
currentdir = os.getcwd()

class hyperparameters:
   def __init__(self):
      self.condition_dict = {"vacant":0, "wall":1, "exit":2, "agent":3}
      self.t_period = 500
      self.t_observe = 0
      self.ep_termination = 50
      self.width = 18
      self.height = 15
      self.agent_eyesight_whole = 7 #odd number
      self.random_closing = True
      self.history_length = 10
      self.bare_intr_coef = 0.5
      
      self.leaking_rate = 0.8
      self.gamma = 0.95
      self.reservoir_size = 1024
      self.sparsity = 1.0 - 0.1 
      self.edge_sparsity = 1.0 - 0.2
      self.central_sparsity = 1.0 - 0.4 
      self.spectral_radius=0.95 
      self.beta = 0.0001 #Ridge term
      self.epsilon = 1.0 #epsilon-greedy
      self.epsilon_decay = 0.02
      self.epsilon_min = 0.02 
      self.decay_ratio = 0.98 #decay of memory matrices per one update of W_out

      self.sigma_W_in_a = 2.0 #stddev of W_in_a


class each_button():
   def __init__(self, frame, text, iy, ix, height, width):
      if (ix>=3 and ix<width-3) and (iy>=3 and iy<height-3) :
         self.button = tk.Button(frame, text=text, bg='black', width=2, height=1, command=self.turn_color)
         self.color_state = np.array([0,0,0])
      else:
         self.button = tk.Button(frame, text=text, bg='#333333', width=2, height=1, command=None)
         self.color_state = np.array([0,0,0])
      self.button.grid(column=ix, row=iy)


   def turn_color(self):
      if self.button["bg"] == "black":
         self.button["bg"] = "red"
         self.color_state = np.array([1,0,0])
      elif self.button["bg"] == "red":
         self.button["bg"] = "green"
         self.color_state = np.array([0,1,0])
      elif self.button["bg"] == "green":
         self.button["bg"] = "blue"
         self.color_state = np.array([0,0,1])
      elif self.button["bg"] == "blue":
         self.button["bg"] = "black"
         self.color_state = np.array([0,0,0])


class main_window(Env):
   def __init__(self, hyperparameters):
      super().__init__(hyperparameters, experience_sharing='on')
      self.window = tk.Tk()
      self.window.title("Room Evacuation ver_0.1.0") 
      self.frame = tk.Frame(self.window)
      self.frame.grid(column=0, row=0)

      self.rows = hyperparameters.height
      self.cols = hyperparameters.width
      self.gamma = hyperparameters.gamma
      self.epsilon_decay = hyperparameters.epsilon_decay
      self.t_period = hyperparameters.t_period
      self.ep_termination = hyperparameters.ep_termination

      self.label_width = 15
      self.txtbox_width = 20
      self.btn = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

      self.state_0 = np.zeros((self.height,self.width,3)).astype(int)

      #self.epsilon_decay_entry = tk.DoubleVar()

      for i_col in range(self.width):
         for i_row in range(self.height):
            self.btn[i_row][i_col] = tk.Button(self.frame)
      for i_col in range(self.width):
         for i_row in range(self.height):
            text_tmp = str(i_row) + ',' + str(i_col)
            self.btn[i_row][i_col] = each_button(self.frame, text=text_tmp, iy=i_row,ix=i_col, height=self.rows, width=self.cols) 

      self.label_frame = tk.LabelFrame(self.frame,text="hyperparameters")
      self.gamma_label = tk.Label(self.label_frame, text="gamma", width=self.label_width)
      self.epsilon_decay_label = tk.Label(self.label_frame, text="epsilon decay", width=self.label_width)
      self.epsilon_min_label = tk.Label(self.label_frame, text="epsilon min", width=self.label_width)
      self.gamma_txtbox = tk.Entry(self.label_frame, width=self.txtbox_width)
      self.epsilon_decay_txtbox = tk.Entry(self.label_frame, width=self.txtbox_width)
      self.epsilon_min_txtbox = tk.Entry(self.label_frame, width=self.txtbox_width)

      self.gamma_txtbox.insert(tk.END, self.gamma)
      self.epsilon_decay_txtbox.insert(tk.END, self.epsilon_decay)
      self.epsilon_min_txtbox.insert(tk.END, self.epsilon_min)

      self.gamma_label.grid(column=0, row=0)
      self.epsilon_decay_label.grid(column=0, row=1)
      self.epsilon_min_label.grid(column=0, row=2)
      self.gamma_txtbox.grid(column=1, row=0)
      self.epsilon_decay_txtbox.grid(column=1, row=1)
      self.epsilon_min_txtbox.grid(column=1, row=2)
      self.label_frame.grid(column=self.cols+1, row=0, columnspan=3, rowspan=6, padx=2, pady=2)

      self.start_btn = tk.Button(self.frame, text='Start', width=10, command=self.get_state)
      self.start_btn.grid(column=self.cols+1,row=self.rows-4, columnspan=3, rowspan=3, padx=2, pady=2)
      self.reset_btn = tk.Button(self.frame, text='Reset all', width=10, command=self.reset_whole)
      self.reset_btn.grid(column=self.cols+1,row=self.rows-2, columnspan=3, rowspan=3, padx=2, pady=2)

      self.conf1_btn = tk.Button(self.frame, text='conf1', width=10, command=self.load_conf1)
      self.conf1_btn.grid(column=self.cols+1,row=self.rows-6, columnspan=1, rowspan=3, padx=2, pady=2)
      self.conf2_btn = tk.Button(self.frame, text='conf2', width=10, command=self.load_conf2)
      self.conf2_btn.grid(column=self.cols+2,row=self.rows-6, columnspan=1, rowspan=3, padx=2, pady=2)
      self.conf3_btn = tk.Button(self.frame, text='conf3', width=10, command=self.load_conf3)
      self.conf3_btn.grid(column=self.cols+3,row=self.rows-6, columnspan=1, rowspan=3, padx=2, pady=2)

      self.global_n_episode = 0
      self.filename_LC = "Learning_Curve.csv"
      self.file_LC = open(self.filename_LC, "w")
      """
      #test of array updating
      self.new_array = np.zeros((self.height,self.width)).astype(int)
      self.update_test = tk.Button(self.frame, text='update pedestrian', width=10, command=self.button_update)
      self.update_test.grid(column=self.width+1,row=self.height-6, columnspan=2, rowspan=3, padx=2, pady=2)
      """

   def mainloop(self):
      self.window.mainloop()

   def get_state(self):
      for i_col in range(self.width):
         for i_row in range(self.height):
            self.state_0[i_row][i_col] = self.btn[i_row][i_col].color_state
      g_temp = float( self.gamma_txtbox.get() )
      ed_temp = float( self.epsilon_decay_txtbox.get() )
      em_temp = float( self.epsilon_min_txtbox.get() )
      if self.check_0to1([g_temp,ed_temp,em_temp]):
         self.gamma = g_temp
         self.epsilon_decay = ed_temp
         self.epsilon_min = em_temp
      self.init_agent_pos = self.state_0[:,:,0].T
      self.init_wall_pos = self.state_0[:,:,1].T
      self.init_exit_pos = self.state_0[:,:,2].T
      self.n_agent = np.sum(self.init_agent_pos)
      self.done_agent_list = [False for _ in range(self.n_agent)]
      self.done_time_list = self.hyperparameters.t_period*np.ones(self.n_agent)

      agent_pos_temp = np.where(self.init_agent_pos==1)
      self.init_agent_pos_list = np.array(agent_pos_temp).T.tolist()
      wall_pos_temp = np.where(self.init_wall_pos==1)
      self.init_wall_pos_list = np.array(wall_pos_temp).T.tolist()
      exit_pos_temp = np.where(self.init_exit_pos==1)
      self.init_exit_pos_list = np.array(exit_pos_temp).T.tolist()

      for i_col in range(self.width):
         for i_row in range(self.height):
            self.inactivate([i_row, i_col]) 
      self.conf1_btn["state"] = tk.DISABLED
      self.conf2_btn["state"] = tk.DISABLED
      self.conf3_btn["state"] = tk.DISABLED
      for i_agent in range(self.n_agent):
         pos_temp_agent = self.init_agent_pos_list[i_agent]
         if i_agent == 0:
            new_agent = Agent(pos_temp_agent[0], pos_temp_agent[1], self.condition, kind='agent', hyperparameters=self.hyperparameters, gamma=self.gamma, epsilon_decay=self.epsilon_decay)
         else:
            new_agent = copy.deepcopy(self.agents[0])
            new_agent.pos = pos_temp_agent
         self.agents.append(new_agent)
      self.run()

   def build_wall(self):
      self.walls = [] 
      self.exits = [] 
      for pos in self.init_wall_pos_list :
         self.set_wall(pos[0],pos[1])
      for pos in self.init_exit_pos_list : 
         self.set_exit(pos[0],pos[1])
         
   def check_0to1(self,x):
      #x : List
      for x_element in x:
         if x_element < 0.0 or x_element > 1.0:
            messagebox.showerror('gamma and epsilon should be larger than 0 and smaller than 1')
            return False
            break
      return True

   def button_update(self):
      for i_col in range(self.width):
         for i_row in range(self.height):
            if self.btn[i_row][i_col].button["bg"] == "black" and self.new_array[i_row][i_col] == 1 :
               self.btn[i_row][i_col].button["bg"] = "red" 
               self.btn[i_row][i_col].color_state = np.array([1,0,0])
            elif self.btn[i_row][i_col].button["bg"] == "red" and self.new_array[i_row][i_col] == 0 :
               self.btn[i_row][i_col].button["bg"] = "black" 
               self.btn[i_row][i_col].color_state = np.array([0,0,0])

      self.update_array() #test of array updating

   def inactivate(self, pos):
      self.btn[pos[0]][pos[1]].button["state"] = tk.DISABLED

   def activate(self, pos):
      self.btn[pos[0]][pos[1]].button["state"] = tk.NORMAL
      if (pos[1]>=3 and pos[1]<self.width-3) and (pos[0]>=3 and pos[0]<self.height-3) :
         self.btn[pos[0]][pos[1]].button["bg"] = 'black'
         self.btn[pos[0]][pos[1]].color_state = np.array([0,0,0])

   def run(self):
      for n_episode in range (self.ep_termination):
         self.reset()
         if (self.global_n_episode+1) in {1,5} or (n_episode+1) == self.ep_termination : 
            filename_gif1 = "animation_" + str(self.global_n_episode+1) + "th_episode.gif"
            self._rendermode_change_to_rgb()
         else:
            self.render_mode = 'off'
         for t1 in range (self.t_period):
            self._render()
            if t1 == self.t_period-1 or all( self.done_agent_list ) :
               self._render()
               self.done = True
            self._step(t1)
            if self.done:
               break
         if self.render_mode == 'rgb_array':
            self.Image_list[0].save(filename_gif1, save_all=True, append_images=self.Image_list[1:], optimize=False, duration=200, loop=0)   
         reward_list = np.zeros(self.n_agent)
         intr_reward_list = np.zeros(self.n_agent)
         for i1 in range (self.n_agent):         
            reward_list[i1] = self.agents[i1].total_reward_extr
            intr_reward_list[i1] = self.agents[i1].total_reward_intr
         if self.global_n_episode == 0 :
            print( 'max', 'mean', 'median', 'evacuated', sep='	' )
            print( 'episode', 'max', 'mean', 'median', 'evacuated', sep=',', file=self.file_LC )
         print( np.max(self.done_time_list), np.mean(self.done_time_list), np.median(self.done_time_list), self.done_agent_list.count(True), sep='	' )
         print( self.global_n_episode+1, np.max(self.done_time_list), np.mean(self.done_time_list), np.median(self.done_time_list), self.done_agent_list.count(True), sep=',', file=self.file_LC )
         self.global_n_episode += 1

   def reset_whole(self):
      np.random.seed(seed)
      self.agents = []
      self.walls = []
      self.exits = []
      self.global_n_episode = 0
      self.test_mode = False
      for i_col in range(self.width):
         for i_row in range(self.height):
            self.activate([i_row, i_col]) 
      self.conf1_btn["state"] = tk.NORMAL
      self.conf2_btn["state"] = tk.NORMAL
      self.conf3_btn["state"] = tk.NORMAL
      self.file_LC.close()
      self.file_LC = open(self.filename_LC, "w")

   """
   #test of array updating
   def update_array(self):
      i_col = np.random.randint(0,self.width,None)
      i_row = np.random.randint(0,self.height,None)
      if self.new_array[i_row][i_col] == 0 :
         self.new_array[i_row][i_col] = 1
   """

   def load_conf1(self) :
      for i_col in range (3,self.width-3):
         for i_row in range (3,self.height-3):
            if (i_row == 3 or i_row == self.height - 4) or (i_col == 3 or i_col == self.width - 4) :
               if i_row == 3 and i_col == (self.width-1)//2 :
                  self.btn[i_row][i_col].button["bg"] = "blue"
                  self.btn[i_row][i_col].color_state = np.array([0,0,1])
               else:
                  self.btn[i_row][i_col].button["bg"] = "green"
                  self.btn[i_row][i_col].color_state = np.array([0,1,0])
            elif i_row == self.height - 5 and i_col%3 == 0 :
               self.btn[i_row][i_col].button["bg"] = "red"
               self.btn[i_row][i_col].color_state = np.array([1,0,0])
            else:
               self.btn[i_row][i_col].button["bg"] = "black"
               self.btn[i_row][i_col].color_state = np.array([0,0,0])

   def load_conf2(self) :
      for i_col in range (3,self.width-3):
         for i_row in range (3,self.height-3):
            if (i_row == 3 or i_row == self.height - 4) or (i_col == 3 or i_col == self.width - 4) :
               if i_row == 3 and i_col == (self.width-1)//2 :
                  self.btn[i_row][i_col].button["bg"] = "blue"
                  self.btn[i_row][i_col].color_state = np.array([0,0,1])
               else:
                  self.btn[i_row][i_col].button["bg"] = "green"
                  self.btn[i_row][i_col].color_state = np.array([0,1,0])
            elif (i_col >= 6 and i_col < self.width - 6) and (i_row == 6 or i_row == 7) :
                  self.btn[i_row][i_col].button["bg"] = "green"
                  self.btn[i_row][i_col].color_state = np.array([0,1,0])
            elif i_row == self.height - 5 and i_col%3 == 0 :
               self.btn[i_row][i_col].button["bg"] = "red"
               self.btn[i_row][i_col].color_state = np.array([1,0,0])
            else:
               self.btn[i_row][i_col].button["bg"] = "black"
               self.btn[i_row][i_col].color_state = np.array([0,0,0])

   def load_conf3(self) :
      list_green = [[3,i_c] for i_c in range(5,self.width-3)] + [[self.height-4,i_c] for i_c in range(3,self.width-3)] 
      list_green += [[7,i_c] for i_c in range(self.width-8,self.width-4)] 
      list_green += [[i_r,3] for i_r in range(6,self.height-4)] + [[i_r,self.width -4] for i_r in range(6,self.height-4)]
      list_green += [[4,5],[4,self.width-4],[5,5],[6,4],[6,5],[6,7],[6,8],[7,7],[7,8],[self.height-5,7],[self.height-6,7],[self.height-6,9],[self.height-6,10],[self.height-6,self.width-6],[self.height-5,self.width-6]]
      list_red = [[self.height-5,4],[self.height-5,6],[self.height-5,10],[self.height-7,10],[self.height-5,self.width-5]]
      for i_col in range (3,self.width-3):
         for i_row in range (3,self.height-3):
            self.btn[i_row][i_col].button["bg"] = "black"
            self.btn[i_row][i_col].color_state = np.array([0,0,0])
      for pos in list_green:
         self.btn[pos[0]][pos[1]].button["bg"] = "green"
         self.btn[pos[0]][pos[1]].color_state = np.array([0,1,0])
      for pos in list_red:
         self.btn[pos[0]][pos[1]].button["bg"] = "red"
         self.btn[pos[0]][pos[1]].color_state = np.array([1,0,0])
      self.btn[5][self.width-4].button["bg"] = "blue"
      self.btn[5][self.width-4].color_state = np.array([0,0,1])

if __name__ == '__main__':
   hyper = hyperparameters()
   window1 = main_window(hyper)

   window1.mainloop()
   window1.file_LC.close()
   """
   print(window1.state_0[:,:,0])
   print(window1.state_0[:,:,1])
   print(window1.state_0[:,:,2])
   print(window1.gamma, window1.epsilon_decay, window1.epsilon_min)
   """