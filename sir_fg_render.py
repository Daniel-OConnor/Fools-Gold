import pygame
import numpy as np
from sir_fg import *

class Visual_SIR_Sim(SIR_Sim):

    def render(self, display_size:int=500):
        assert (self.total_steps != -1)

        pygame.init()
        screen = pygame.display.set_mode((display_size, display_size))
        pygame.display.set_caption("Visual_SIR_Sim")

        LIGHT_GREY = (200,200,200)
        screen.fill(LIGHT_GREY)
        c = int(0.02*display_size) # circumference of circles

        COLOURS = [(50,200,50), (255,50,50), (100,100,100)]
        s = 0
        print("total number of steps should be", self.total_steps)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if s <= self.total_steps:
                latent_step = self.latents[s]
                print(s)
                for i in range(self.pop):
                    p_state = int(latent_step[3*i+2])
                    p_pos = (latent_step[3*i], latent_step[3*i+1])
                    screen_pos = self.posToScreen(p_pos,display_size)
                    pygame.draw.circle(screen, COLOURS[p_state], screen_pos, c)
                    pygame.draw.circle(screen, (0,0,0), screen_pos, c, 2)
                
                pygame.display.update()
                pygame.time.wait(500)
                screen.fill(LIGHT_GREY)
                s += 1



        pygame.quit()

    # internal function to convert from person position to screen position
    def posToScreen(self, pos, DS:int):
        return (int(pos[0]*DS/self.city_size), int(pos[1]*DS/self.city_size))

s = Visual_SIR_Sim(pop=75, city_size=10);
latentData = s.run();
s.render(750)
print(latentData)


