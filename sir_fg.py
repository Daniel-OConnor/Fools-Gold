import numpy as np
import random

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
        self.velocity = self.accel = np.zeros(2);
        self.city_size = city_size;
        self.speed = speed;
    
    def update_position(self):
        force = np.zeros(2)
        for i in range(2):
            force[i] = (random.uniform(-self.speed,self.speed) + self.accel[i])/2.0
            self.velocity[i] += force[i]
        self.accel = force
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

class SIR_Sim():

    def __init__(self, pop:int=20, city_size=5,
            infection_rad=2, p_infection_per_day=0.3,
            infection_duration=14, speed=0.1):
        self.pop = pop; self.city_size = city_size;
        self.infection_rad = infection_rad; self.speed = speed;
        self.p_infection_per_day = p_infection_per_day;
        self.infection_duration = infection_duration;

        self.latents = []

        self.people = []
        self.add_people()

        self.update_latents()

        if (TEST): print([list(p.pos) for p in self.people], [p.state for p in self.people])

        self.num_S = pop; self.num_I = self.num_R = 0;

        self.step = 0
        self.total_steps = -1 # placeholder value until simulator is run
    
    def add_people(self):
        for i in range(self.pop):
            rand_pos = np.array([random.random(),random.random()])*self.city_size
            self.people.append(Person(i,rand_pos,self.city_size,self.speed))

    def run(self, steps=0):
        self.infect_patient_zero()
        while (self.step < steps or (steps==0 and self.num_I > 0)):
            self.update()
            if (TEST): print([list(p.pos) for p in self.people], [p.state for p in self.people])
        print("simulation complete; latent variable values returned")
        self.total_steps = self.step;
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
        latent_step = np.zeros(self.pop*3)
        for p in self.people:
            j = p.id
            latent_step[3*j] = p.pos[0]
            latent_step[3*j+1] = p.pos[1]
            latent_step[3*j+2] = p.state

        self.latents.append(latent_step)







s = SIR_Sim();
latentData = s.run()
print(latentData)