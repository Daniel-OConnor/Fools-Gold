import torch
import numpy as np
import random
from simulator import ProbSimulator

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
        force = np.zeros(2)
        for i in range(2):
            force[i] = random.uniform(-self.speed,self.speed)
            self.velocity[i] += force[i]

        norm = np.linalg.norm(self.velocity)
        if (norm > self.max_speed):
            self.velocity = self.velocity/norm
            # (normalize to 1 to prevent exceeding max_speed)

        for i in range(2): # for x/y-axis
            new_pos_i = self.pos[i]+self.velocity[i] # pre-compute new position
            if (new_pos_i < 0 or new_pos_i > self.city_size): # if would move person beyond city limits
                self.velocity[i] *= -1 # approximate collision with city boundary
        self.time += 1
        self.pos = self.pos + self.velocity # can now 'safely' update position

    def update_state(self, state:int):
        self.state = state
        if (state==I): self.time_infected=self.time;

class SIR_Sim(ProbSimulator):

    # required by Simulator class
    # x_size is size of output (proprtion infected and/or recovered?)
    # θ_size is size of sim. run parameters (infection radius, duration and prob. catching per day)
    x_size = 1; θ_size = 3;

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
        
        self.infect_patient_zero()
        # initial latent vars update stores initial positions, velocities and patient zero state
        self.update_latents()
        while (self.step < steps or (steps==0 and self.num_I > 0)):
            self.update()
            if (TEST): print([list(p.pos) for p in self.people], [p.state for p in self.people])
        print("simulation complete; latent variable values returned")
        self.total_steps = self.step;
        return self.torch_latents(self.latents)
    
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
        for p in self.people:
            p.update_position()

        self.update_states()
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
        latent_step = np.zeros(self.pop*5)
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

    def p(self, zs, θ):
        """
        Calculate conditional probabilities for a run of the simulator
        
        Arguments:
            zs:        List[torch.Tensor], latent variables
            θ:         torch.Tensor, parameters
        Returns:
            ps: torch.Tensor, where ps[i] = p(z_i|θ, zs[:i])
        """
        ps = torch.zeros(len(zs))
        ps[0] = ps[-1] = 1
        # (probability of init. state & final latents given previous latents is 1)

        for i in range(1, len(zs) - 1):
            ps[i] = p_latent_step(zs[i], zs[i-1])

        return ps
    
    def p_latent_step(self,zi,zprev,θ):
        # zprev is the previous step's latent variables i.e. z_i-1
        p_zi_cond = 1 # initialise p(z_i|θ, z_i-1)
        
        # probability calculation goes here

        return p_zi_cond







s = SIR_Sim();
latentData = s.simulate([0.3,2,14])
print(latentData)