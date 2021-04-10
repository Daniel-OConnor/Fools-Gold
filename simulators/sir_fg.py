import torch
import torch.distributions.uniform as torchUni
from torch.distributions.binomial import Binomial
from scipy.stats import binom
import numpy as np
import random
from .simulator import ProbSimulator
import math

# susceptible, infected, recovered/removed
S = 0; I = 1; R = 2
TEST = False;
random.seed()

class Person():

    def __init__(self, id, pos, city_size, speed):
        self.id = id # unique person identifier
        self.state = S; self.pos = pos
        self.infection_radius = 0.5; #self.symptomatic = False
        self.max_speed = 0.25; self.time = 0;
        self.time_infected = 0;
        self.velocity = np.zeros(2);
        self.city_size = city_size;
        self.speed = speed;
    
    def update_position(self):
        # update position using previous pos and velocity
        self.pos = self.pos + self.velocity

        # now calculate new velocity ready for next step
        force = np.zeros(2)
        for i in range(2):
            force[i] = random.uniform(-self.speed,self.speed)
            self.velocity[i] += force[i]
            # prevent exceeding max speed along the axis:
            if (abs(self.velocity[i]) > self.max_speed):
                self.velocity[i] = self.max_speed * (1 if self.velocity[i]>=self.max_speed else -1)

        for i in range(2): # for x/y-axis
            new_pos_i = self.pos[i]+self.velocity[i] # pre-compute new position
            if (new_pos_i < 0 or new_pos_i > self.city_size): # if would move person beyond city limits
                self.velocity[i] *= -1 # approximate collision with city boundary
        self.time += 1

    def update_state(self, state:int):
        self.state = state
        if (state==I): self.time_infected=self.time;

class SIR_Sim(ProbSimulator):

    # required by Simulator class
    # x_size is size of output (proprtion infected and/or recovered?)
    # θ_size is size of sim. run parameters (infection radius, duration and prob. catching per day)
    x_size = 1; theta_size = 3;

    def __init__(self, pop:int=20, city_size=5, speed=0.1):
        self.pop = pop;
        self.city_size = city_size;
        self.speed = speed;

        # THETA PARAMETERS
        # p_infection_per_day (=0.3)
        # infection_rad (=2)
        # infection_duration (=14)
        # ... are now given in θ argument to .simulate()

        self.latents = []

        self.people = []
        self.add_people()

        if (TEST): print([list(p.pos) for p in self.people], [p.state for p in self.people])

        self.num_S = pop; self.num_I = self.num_R = 0;

        self.step = 0
        self.total_steps = -1 # placeholder value until simulator is run
    
    def add_people(self):
        for i in range(self.pop):
            rand_pos = np.array([random.random(),random.random()])*self.city_size
            self.people.append(Person(i,rand_pos,self.city_size,self.speed))

    def simulate(self, θ, steps=0):
        self.p_infection_per_day = θ[0]
        self.infection_rad = θ[1]
        self.infection_duration = θ[2]
        self.latents = []
        self.num_S = self.pop
        self.num_I = self.num_R = 0
        self.people = []
        self.add_people()
        
        self.infect_patient_zero()
        # initial latent vars update stores initial positions, velocities and patient zero state
        self.update_latents()
        while (self.step < steps or (steps==0 and self.num_I > 0)):
            self.update()
            if (TEST): print([list(p.pos) for p in self.people], [p.state for p in self.people])
        self.total_steps = self.step
        self.latents.append(torch.tensor([(self.num_I+self.num_R)/self.pop]))
        return self.latents
    
    def infect_patient_zero(self):
        targetID:int = random.randrange(self.pop)
        target:Person = self.people[targetID]
        self.infect(target)

    def infect(self, person:Person):
        if (person.state==S):
            person.update_state(I) # infects target
            self.num_I += 1; self.num_S -= 1
        else:
            pass; # can't infect bc infected or recovered/removed
    
    def recover(self, person:Person):
        if (person.state==I):
            person.update_state(R) # target recovers from infection
            self.num_I -= 1; self.num_R += 1
        else:
            pass; # can't recover bc already recovered or susceptible

    def update(self):
        self.update_states()
        
        for p in self.people:
            p.update_position()

        self.step += 1
        self.update_latents()
    
    def update_states(self):
        s_group, i_group = [
            list(filter(
                    lambda m: m.state == state,
                    self.people
            ))
            for state in [S, I]
        ]
        for s_person in s_group:
            for i_person in i_group:
                dist = np.linalg.norm(i_person.pos - s_person.pos)
                if (dist < s_person.infection_radius and random.random() < self.p_infection_per_day):
                    self.infect(s_person)
        for i_person in i_group:
            if (i_person.time - i_person.time_infected) > self.infection_duration:
                self.recover(i_person)
    
    def update_latents(self):
        latent_step = torch.zeros(self.pop*5)
        for p in self.people:
            j = p.id
            latent_step[5*j] = p.pos[0]
            latent_step[5*j+1] = p.pos[1]
            latent_step[5*j+2] = p.velocity[0]
            latent_step[5*j+3] = p.velocity[1]
            latent_step[5*j+4] = p.state

        self.latents.append(latent_step)
    
    def torch_latents(self, latent_data):
        for l in latent_data:
            l = torch.from_numpy(l)
        return torch.tensor(latent_data)

    def log_p(self, zs, θ):
        """
        Calculate conditional probabilities for a run of the simulator
        
        Arguments:
            zs:        List[torch.Tensor], latent variables
            θ:         torch.Tensor, parameters
        Returns:
            ps: torch.Tensor, where ps[i] = p(z_i|θ, zs[:i])
        """
        ps = [0 for _ in range(len(zs)-1)]

        # === PROB. DENSITY CALCULATION FOR ps[0] ===
        
        # torch Uniform distribution for initial positions of population
        uni_dist = torchUni.Uniform(0, self.city_size)

        ps[0] = torch.tensor(1)
        # computes ps[0] as product of pdfs of independent locations of each person
        for i in range(self.pop):
            ps[0] = ps[0] + uni_dist.log_prob(zs[0][5*i])  # initial pos x-coord
            ps[0] = ps[0] + uni_dist.log_prob(zs[0][5*i+1])  # initial pos y-coord
        # then multiply by probability of person[i] being chosen as patient zero
        ps[0] = ps[0] + math.log(1.0/float(self.pop))
        for i in range(1, len(zs)-1):
            ps[i] = self.p_latent_step(zs[i], zs[i-1], θ)
        return sum(ps)
    
    def p_latent_step(self,zi,zprev,θ):
        # zprev is the previous step's latent variables i.e. z_i-1
        log_p_zi_cond = torch.tensor(0., requires_grad=True) # initialise p(z_i|θ, z_i-1)
        
        # === LATENT POSITION/VELOCITY FACTORS ===
        # p(pos | prev pos, prev vel) = 1, so ignore
        # p(vel | prev pos, prev vel) = uniform density sample (but also consider max speed)
        uni_dist = torchUni.Uniform(-self.speed, self.speed)
        """for j in range(self.pop):
            velX = zi[5*j+2]; velY = zi[5*j+3] # extract velocity in x/y from latent step
            velX_prev = zprev[5*j+2]; velY_prev = zprev[5*j+3] # extract previous latent step velocities
            for (v,u) in [(velX, velX_prev), (velY,velY_prev)]:
                
                if (abs(v) < self.people[0].max_speed): # fixed at 0.25 for all people
                    log_p_zi_cond = log_p_zi_cond + uni_dist.log_prob(abs(abs(v)-abs(u))) # abs(v)-abs(u) is the force that must've been applied
                    # i.e. multiply by pdf evalutated at uniform-sampled value from this step
                
                else: # then v == 0.25 or -0.25, SPECIAL CASE
                    # special since multiple uniform 'force' samples could've yielded this velocity value
                    # because computed velocities >0.25 get capped at 0.25 (max_speed)
                    min_force = abs(abs(v)-abs(u)) # minimum force that was applied to get v==(-)0.25
                    p_v_given_u = ((self.speed - min_force)/self.speed)
                    # update with nat.log of uniform probability that such a force was sampled
                    log_p_zi_cond = log_p_zi_cond + torch.log(p_v_given_u)"""
        
        # === LATENT INFECTION STATE FACTORS ===
        # p(state | previous latents) is deterministic in all cases except:
        # ...p(state==I | previous state==S, other latents)
        # ...p(state==S | previous state==S, other latents => infectious person within radius)
        states_curr = [zi[5 * j + 4] for j in range(len(self.people))]
        states_prev = [zprev[5 * j + 4] for j in range(len(self.people))]
        for j in range(len(self.people)):
            if (states_prev[j] == 0):  # if S -> I or S -> S
                num_infected_in_range = 0
                for k in range(len(self.people)):
                    if (states_prev[k] == 1):  # if k is index of an infected person
                        # check if within infection radius
                        dist = np.linalg.norm([zprev[5 * j] - zprev[5 * k], zprev[5 * j + 1] - zprev[5 * k + 1]])
                        if (dist <= θ[1]):
                            num_infected_in_range += 1
                if num_infected_in_range > 0:
                    # evaluate p(gets infected | some nearby infected people)
                    p_inf = self.p_infected(num_infected_in_range, θ[0])
                    if (states_curr[j]==1): # if person j DID get infected...
                        log_p_zi_cond = log_p_zi_cond + p_inf
                    else: # if person j DIDN'T get infected...
                        log_p_zi_cond = log_p_zi_cond + torch.log(1 - torch.exp(p_inf))
            None
        return log_p_zi_cond
    
    # internal function - evalutates p(gets infected | some nearby infected people)
    def p_infected(self, n, p):
        binom_dist = Binomial(n, p)
        # returns 1 - p(0 transmissions to susceptible | 'n' nearby infected, 'p' prob. of transmission)
        p0 = binom_dist.log_prob(torch.tensor(0.))
        out = torch.log(1 - torch.exp(p0))
        return out


#s = SIR_Sim();
#latentData = s.simulate([0.3,2,14])
#print(latentData)
