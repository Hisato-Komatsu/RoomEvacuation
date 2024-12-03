"""
reservoir+LSPI
1/12 避難経路シミュレーション用に、各エージェントに変数doneとcountを実装。doneがTrueになったらlast_contributionを計算した後countをFalseにし、以降の時刻で各計算から除外。
1/13 exitに達したエージェントを環境から取り除く処理を実装、プロトタイプ作成
1/15 上下左右の移動の他に、「動かない」という選択肢を実装。
1/16 前後の扉の内一方が半々の確率で閉じる。
1/17 座標の情報を入力。また、学習曲線に「避難完了したエージェントの数」も追加
3/25 modified the calculation method of matrix A
3/25 added intrinsic reward
3/26 epsilonがepsilon_minよりも小さくなったらepsilon_minに戻すように変更
4/2 test mode導入
4/18 exit用のチャンネルの入力重みをdenseにすることを検討...Cut_probのch3を0.0にするコードを作成
12/2 intrinsic rewardを、単に各マスを訪れた回数に比例した負の値に変更
"""
import numpy as np
#import gc

import copy
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
#from collections import deque


class Env:
   def __init__(self, hyperparameters, render_mode='off', experience_sharing='on'):
      # render_mode : 'rgb_array' or 'off'
      self.render_mode = render_mode
      # experience_sharing : 'on' or 'off'
      self.experience_sharing = experience_sharing
      if self.render_mode == 'rgb_array' :
         self.Image_list = []

      self.agents = []
      self.walls = []
      self.exits = []
      self.hyperparameters = hyperparameters
      self.condition_dict = self.hyperparameters.condition_dict
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.area = self.width*self.height
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,3))
      self.view_vx = np.zeros((self.width,self.height))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.pos_to_object = [[None for i in range(self.height)] for j in range(self.width)]
      self.done = False
      self.epsilon_min = self.hyperparameters.epsilon_min
      self.random_closing = self.hyperparameters.random_closing

      self.gamma_intr = self.hyperparameters.gamma_intr
      self.visit_count = np.zeros((self.width,self.height))

      self.test_mode = False


   def agent_position(self, i_agent):
      if i_agent == self.n_agent-1:
         return np.array([(self.x_min+self.x_max)//2,self.y_min])
      else:
         return np.array([self.x_min+2+(i_agent%10)+((i_agent%10)//5),self.y_min+4+2*(i_agent//10)])

   def set_wall(self,x,y):
      new_wall = Object(x,y, self.condition, kind=("wall"), hyperparameters=self.hyperparameters)
      self.walls.append(new_wall)
      self.pos_to_object[x][y] = new_wall
      self.view[x][y][1] = new_wall.color

   def set_exit(self,x,y):
      new_exit = Object(x,y, self.condition, kind=("exit"), hyperparameters=self.hyperparameters)
      self.exits.append(new_exit)
      self.pos_to_object[x][y] = new_exit
      self.view[x][y][2] = new_exit.color

   def reset(self):
      #reset of environment with keeping the result of training
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,3))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.pos_to_object = [[None for i in range(self.height)] for j in range(self.width)]
      self.done = False
      self.done_agent_list = [False for _ in range(self.n_agent)]
      self.done_time_list = self.hyperparameters.t_period*np.ones(self.n_agent)
      self.build_wall()
      for i_agent in range(self.n_agent):
         pos_temp_agent = np.array( self.init_agent_pos_list[i_agent] )
         self.agents[i_agent].pos = pos_temp_agent
         self.agents[i_agent].reset()
         self.pos_to_object[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.agents[i_agent]
         self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.condition_dict["agent"]
         self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color

   def _step(self, t):
      self.view_vx = np.zeros((self.width,self.height))
      #agents' action
      for i_agent in range(self.n_agent):
         if self.agents[i_agent].count :
            action=[0,0]
            #self.agents[i_agent]._key_step(self.pos_to_object)
            
            if t >= 1:
               self.agents[i_agent].reward_past = self.agents[i_agent].reward_extr
               self.agents[i_agent].action_past = action
               self.agents[i_agent].Xi_temp_part0 = copy.deepcopy(self.agents[i_agent].Xi_temp_part1)

            delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
            dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
            state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
            self.agents[i_agent].action, self.agents[i_agent].Xi_temp_part1, self.agents[i_agent].Q_temp = self.agents[i_agent].choose_action(state)
            self.agents[i_agent]._step(self.agents[i_agent].action, self.pos_to_object)
            self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] += 1     
      
            #intrinsic reward
            self.visit_count[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] += 1.0
            self.agents[i_agent].reward_intr_past = - self.agents[i_agent].coef_intr*self.visit_count[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]]/np.sum(self.visit_count)

      #位置の更新はEnv側で一括管理した方が混線しにくい。
      for i_agent in range(self.n_agent):
         if self.agents[i_agent].count :
            if self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] in {self.condition_dict["vacant"], self.condition_dict["exit"]} and self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == 1 :
               self.agents[i_agent].move_permission = 'yes'

      for i_agent in range(self.n_agent): 
         #self.agents[i].countという変数を作り、これがTrueの時だけ勘定に入れる。self.agents[i].countはlast_contributionを計算した後Falseにする。
         if self.agents[i_agent].count :
            #update of the position of agents
            if self.agents[i_agent].move_permission == 'yes' :
               self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.condition_dict["vacant"]
               self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = 0.0
               self.pos_to_object[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = None
               if self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == self.condition_dict["vacant"]:
                  self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] = self.condition_dict[self.agents[i_agent].kind]
                  self.view[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]][0] = self.agents[i_agent].color
                  self.view_vx[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.agents[i_agent].new_pos[0] - self.agents[i_agent].pos[0]
                  self.pos_to_object[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] = self.agents[i_agent]
                  self.agents[i_agent].reward_extr = 0.0
                  self.agents[i_agent].pos = self.agents[i_agent].new_pos
               elif self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == self.condition_dict["exit"]:
                  self.agents[i_agent].reward_extr = 1.0
                  self.view_vx[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.agents[i_agent].new_pos[0] - self.agents[i_agent].pos[0]
                  self.agents[i_agent].pos = self.agents[i_agent].new_pos
                  self.agents[i_agent].done = True
                  self.done_agent_list[i_agent] = True
                  self.done_time_list[i_agent] = float(t+1)
            else:
               self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
               self.agents[i_agent].reward_extr = 0.0
   
            self.agents[i_agent].total_reward_extr += self.agents[i_agent].reward_extr
            self.agents[i_agent].total_reward_intr += self.agents[i_agent].reward_intr_past
            self.agents[i_agent].move_permission = 'no'

            #update of memory matrices
            if t >= 1 :
               
               Xi_temp = self.agents[i_agent].Xi_temp_part0 - self.gamma*self.agents[i_agent].Xi_temp_part1
               Xi_temp_intr = self.agents[i_agent].Xi_temp_part0 - self.gamma_intr*self.agents[i_agent].Xi_temp_part1

               if self.experience_sharing == 'on':
                  self.agents[0].reward_list.append(self.agents[i_agent].reward_past)
                  self.agents[0].Xi_temp_list.append(Xi_temp)
                  self.agents[0].Xi_temp_intr_list.append(Xi_temp_intr)
                  self.agents[0].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0)

                  self.agents[0].reward_intr_list.append(self.agents[i_agent].reward_intr_past)

               elif self.experience_sharing == 'off':
                  self.agents[i_agent].reward_list.append(self.agents[i_agent].reward_past)
                  self.agents[i_agent].Xi_temp_list.append(Xi_temp)
                  self.agents[i_agent].Xi_temp_intr_list.append(Xi_temp_intr)
                  self.agents[i_agent].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0)

                  self.agents[i_agent].reward_intr_list.append(self.agents[i_agent].reward_intr_past)


            if self.agents[i_agent].done:
               #calculation on the last step
               # contribution of t that t+n gets the last reward
               delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
               dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
               state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]

               action_past = action
               self.agents[i_agent].Xi_temp_part0 = copy.deepcopy(self.agents[i_agent].Xi_temp_part1)
               self.agents[i_agent].reward_past = self.agents[i_agent].reward_extr
               self.agents[i_agent].action, self.agents[i_agent].Xi_temp_part1, self.agents[i_agent].Q_temp = self.agents[i_agent].choose_action(state)
            
               
               Xi_temp = self.agents[i_agent].Xi_temp_part0 - self.gamma*self.agents[i_agent].Xi_temp_part1
               Xi_temp_intr = self.agents[i_agent].Xi_temp_part0 - self.gamma_intr*self.agents[i_agent].Xi_temp_part1

               if self.experience_sharing == 'on':
                  self.agents[0].reward_list.append(self.agents[i_agent].reward_past)
                  self.agents[0].Xi_temp_list.append(Xi_temp)
                  self.agents[0].Xi_temp_intr_list.append(Xi_temp_intr)
                  self.agents[0].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0) 

                  self.agents[0].reward_intr_list.append(0.0)

               elif self.experience_sharing == 'off':
                  self.agents[i_agent].reward_list.append(self.agents[i_agent].reward_past)
                  self.agents[i_agent].Xi_temp_list.append(Xi_temp)
                  self.agents[i_agent].Xi_temp_intr_list.append(Xi_temp_intr)
                  self.agents[i_agent].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0) 

                  self.agents[i_agent].reward_intr_list.append(0.0)

               if self.experience_sharing == 'on':
                  self.agents[0].reward_list.append(0.0)
                  self.agents[0].Xi_temp_list.append(self.agents[i_agent].Xi_temp_part1)
                  self.agents[0].Xi_temp_intr_list.append(self.agents[i_agent].Xi_temp_part1)
                  self.agents[0].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part1)

                  self.agents[0].reward_intr_list.append(0.0)

               elif self.experience_sharing == 'off':
                  self.agents[i_agent].reward_list.append(0.0)
                  self.agents[i_agent].Xi_temp_list.append(self.agents[i_agent].Xi_temp_part1)
                  self.agents[i_agent].Xi_temp_intr_list.append(self.agents[i_agent].Xi_temp_part1)
                  self.agents[i_agent].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part1)

                  self.agents[i_agent].reward_intr_list.append(0.0)

               self.agents[i_agent].count = False
               
      if not self.test_mode :
         if self.done and self.experience_sharing == 'on':
            self.agents[0].calculate_Wout()
            for i_agent in range(self.n_agent):
               self.agents[i_agent].W_out = copy.deepcopy(self.agents[0].W_out)
               self.agents[i_agent].W_out_intr = copy.deepcopy(self.agents[0].W_out_intr)

         if self.done and self.experience_sharing == 'off':
            for i_agent in range(self.n_agent):
               self.agents[i_agent].calculate_Wout()

         if self.done:
            for i_agent in range(self.n_agent):
               self.agents[i_agent].epsilon = max( self.agents[i_agent].epsilon*self.epsilon_decay, self.epsilon_min )

      else :
         for i_agent in range(self.n_agent):
            self.agents[i_agent].Xi_temp_list = []
            self.agents[i_agent].Xi_temp_intr_list = []
            self.agents[i_agent].Xi_temp_p0_list = []

            self.agents[i_agent].reward_intr_list = []

      #resetting move_candidate_number
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)

   def _render(self):
      if self.render_mode == 'rgb_array' :
         view_3ch = self.view.transpose(1,0,2)
         view_3ch = np.maximum( 255.0*np.minimum( view_3ch, np.ones(view_3ch.shape) ), np.zeros(view_3ch.shape) ).astype(np.uint8)
         self.Image_list.append( Image.fromarray( np.repeat(np.repeat(view_3ch,5,axis=0),5,axis=1) ) )
      elif self.render_mode == 'off' :
         pass

   def _rendermode_change_to_rgb(self):
      self.render_mode = 'rgb_array'
      self.Image_list = []

class Object:
   def __init__(self,x,y,condition, kind, hyperparameters):
      self.hyperparameters = hyperparameters
      self.condition_dict = self.hyperparameters.condition_dict
      self.action_space = [0,0] 
      self.v = 1
      self.vskew = 1
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.kind = kind
      self.define_orientation = {"left":0, "right":1}
      self.orientation = 0
      self.attack_switch = 0     
      self.attack_existence = False
      self.agent_eyesight_whole = hyperparameters.agent_eyesight_whole 
      self.agent_eyesight_1side = self.agent_eyesight_whole//2

      self.attacked_count = 0
      self.attacker = []
      self.move_permission = 'no'

      self.pos = np.array([x,y])
      self.color = 1.0
      condition[self.pos[0]][self.pos[1]] = self.condition_dict[kind]

   def _step(self, action, pos_to_object):
      #action[0]...direction of the movement, action[1]:attacking
      if action == 0:
         direction = np.array([1,0])
      elif action == 1:
         direction = np.array([0,1])
      elif action == 2:
         direction = np.array([-1,0])
      elif action == 3:
         direction = np.array([0,-1])
      elif action == 4:
         direction = np.array([0,0])
      self.new_pos = self.pos + direction
#Periodic boundary condition
      if self.new_pos[0] < 0:
         self.new_pos[0] += self.width
      elif self.new_pos[0] >= self.width:
         self.new_pos[0] -= self.width
      if self.new_pos[1] < 0:
         self.new_pos[1] += self.height
      elif self.new_pos[1] >= self.height:
         self.new_pos[1] -= self.height
#位置の更新はEnv側に運び出した。

class Agent(Object):
   def __init__(self,x,y,condition, kind, hyperparameters):
      super().__init__(x,y,condition, kind, hyperparameters)
      # building neural net
      self.state_size = self.agent_eyesight_whole*self.agent_eyesight_whole*3
      self.action_size = 5
      self.action_choice = [_ for _ in range (self.action_size)] 
      self.sparsity = self.hyperparameters.sparsity
      self.edge_sparsity = self.hyperparameters.edge_sparsity
      self.central_sparsity = self.hyperparameters.central_sparsity
      self.spectral_radius = self.hyperparameters.spectral_radius
      self.lr = self.hyperparameters.leaking_rate
      self.beta = self.hyperparameters.beta # Ridge term 
      self.reservoir_size = self.hyperparameters.reservoir_size
      self.reservoir_output_size = self.reservoir_size + 1
      self.epsilon = self.hyperparameters.epsilon #epsilon-greedy
      self.decay_ratio = self.hyperparameters.decay_ratio #decay of memory matrices per one update of W_out

      #intrinsic reward
      self.coef_intr = self.hyperparameters.coef_intr
      self.decay_ratio_intr = self.hyperparameters.decay_ratio_intr #decay of memory matrices per one update of W_out_intr

      self.reward_list = []
      self.Xi_temp_list = []
      self.Xi_temp_intr_list = []
      self.Xi_temp_p0_list = []

      self.reward_intr_list = []

      self.total_reward = 0.0
      self.total_reward_extr = 0.0
      self.total_reward_intr = 0.0
      self.reward = 0.0
      self.reward_extr = 0.0
      self.reward_intr = 0.0
      self.eaten_foods = 0
      self.done = False
      self.count = True
      self.sigma_W_in_a = self.hyperparameters.sigma_W_in_a

      self.W_in_s = np.random.normal(loc=0.0,scale=1.0,size=(self.reservoir_size,self.state_size+1))
      self.W_in_a = np.random.normal(loc=0.0,scale=self.sigma_W_in_a,size=(self.reservoir_size,self.action_size))
      self.W_res = np.random.normal(loc=0.0,scale=1.0,size=(self.reservoir_size,self.reservoir_size))
      self.W_out = np.random.normal(loc=0.0,scale=0.0,size=(1,self.reservoir_output_size) )
      self.X_esn = np.zeros(self.reservoir_size)

      self.W_out_intr = np.random.normal(loc=0.0,scale=0.0,size=(1,self.reservoir_output_size) )

      Cut_prob_W_in_s = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.state_size+1))
      Cut_prob_W_res = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.reservoir_size))

      Cut_prob_mat = self.edge_sparsity * np.ones( (self.agent_eyesight_whole,self.agent_eyesight_whole,3) )
      Cut_prob_mat[self.agent_eyesight_1side-1:self.agent_eyesight_1side+2,self.agent_eyesight_1side-1:self.agent_eyesight_1side+2,:] = self.central_sparsity
      Cut_prob_mat = np.concatenate( [Cut_prob_mat.reshape(-1), [0.0]] )

      for i_1 in range(self.W_in_s.shape[0]):
         for i_2 in range(self.W_in_s.shape[1]):
            if Cut_prob_W_in_s[i_1][i_2] < Cut_prob_mat[i_2] :
               self.W_in_s[i_1][i_2] = 0.00

      for i_1 in range(self.W_res.shape[0]):
         for i_2 in range(self.W_res.shape[1]):
            if Cut_prob_W_res[i_1][i_2] < self.sparsity:
               self.W_res[i_1][i_2] = 0.00

      specrad_temp_W_res = np.max( abs(np.linalg.eigvals(self.W_res)) )
      self.W_res *= self.spectral_radius/specrad_temp_W_res

      self.W_in_s = csr_matrix(self.W_in_s)
      self.W_res = csr_matrix(self.W_res)

      self.XXT = self.beta*np.identity(self.reservoir_output_size)
      self.rX = np.zeros((1, self.reservoir_output_size))

      #intrinsic reward
      self.XXT_intr = self.beta*np.identity(self.reservoir_output_size) 
      self.rX_intr = np.zeros((1, self.reservoir_output_size))


   #methods related with reinforcement learning
   def _ReLU(self,x):
      zeros = np.zeros(x.shape)
      return np.maximum(x,zeros)

   def _softmax(self,z):
      z_rescale = self.inverse_temperature*(z - np.max(z))
      z_exp = np.exp(np.clip(z_rescale,-250,250))
      return z_exp/( np.sum(z_exp) )

   def choose_action(self, state): #epsilon-greedy
      self.X_in = np.concatenate( [state.reshape(-1), [1.0]] )
      Input_state = np.repeat( ((self.W_in_s*self.X_in) + (self.W_res*self.X_esn)).reshape(-1,1), self.action_size, axis=1)
      self.X_esn_tilde = self._ReLU( Input_state + self.W_in_a )
      X_esn_candidates = np.repeat(self.X_esn.reshape(-1,1), self.action_size, axis=1) + self.lr*(self.X_esn_tilde - np.repeat(self.X_esn.reshape(-1,1), self.action_size, axis=1) )
      self.Q = np.dot(self.W_out,np.concatenate([X_esn_candidates, np.ones((1,self.action_size))],axis=0) ).reshape(-1)
      self.Q_intr = np.dot(self.W_out_intr,np.concatenate([X_esn_candidates, np.ones((1,self.action_size))],axis=0) ).reshape(-1)
      Q_sum = self.Q + self.Q_intr

      p = np.random.uniform(low=0.0,high=1.0)
      if p < self.epsilon or np.max(Q_sum) - np.min(Q_sum) < 1.0e-10 :
         action_chosen = np.random.choice(self.action_choice) 
      else:
         action_chosen = np.argmax(Q_sum) 
      self.X_esn = X_esn_candidates[:,action_chosen]

      return action_chosen, copy.deepcopy( np.concatenate([self.X_esn,[1.0]]) ), self.Q[action_chosen] # returns action and corresponding X_res

   def calculate_Wout(self): 
      self.rX += np.dot(np.array(self.reward_list).reshape(1,-1), np.array(self.Xi_temp_p0_list) )
      self.XXT += np.dot(np.array(self.Xi_temp_list).T , np.array(self.Xi_temp_p0_list) )
      XXTinv = np.linalg.inv( self.XXT )
      self.W_out = np.dot(self.rX,XXTinv)
      self.rX *= self.decay_ratio
      self.XXT *= self.decay_ratio

      self.rX_intr += np.dot(np.array(self.reward_intr_list).reshape(1,-1), np.array(self.Xi_temp_p0_list) )
      self.XXT_intr += np.dot(np.array(self.Xi_temp_intr_list).T , np.array(self.Xi_temp_p0_list) )
      XXTinv_intr = np.linalg.inv( self.XXT_intr )
      self.W_out_intr = np.dot(self.rX_intr,XXTinv_intr)
      self.rX_intr *= self.decay_ratio_intr
      self.XXT_intr *= self.decay_ratio_intr

      self.reward_list = []
      self.Xi_temp_list = []
      self.Xi_temp_intr_list = []
      self.Xi_temp_p0_list = []

      self.reward_intr_list = []
      self.f_targ_list = []
      self.Xi_RND_temp_list = []

   def last_contribution(self, x_dense1):
      X0 = copy.deepcopy(x_dense1).reshape(1,-1) 
      return np.dot(X0.T,X0)

   def reset(self):
      self.X_esn = np.zeros(self.reservoir_size)
      self.total_reward_extr = 0.0
      self.total_reward_intr = 0.0
      self.done = False
      self.count = True
