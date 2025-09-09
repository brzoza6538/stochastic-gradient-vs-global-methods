def_max_fes = 0
def_clamps = [1,9]
def_checkpoints = [0, 1, 2, 3, 4, 5, 6, 7]
def_smallest_val = 1e9
import random
import numpy as np
import math
RETRIES = 25

class NL_SHADE_RSP_MID():
    def __init__(
            self,
            f_objective,
            dimension,
            X=None,
            pop_size=None,
            memory_size=None,
            archive_size=None,
            max_fes=def_max_fes,
            objective_limit=None,
            min_clamp=def_clamps[0],
            max_clamp=def_clamps[1],
            checkpoints=def_checkpoints,
            smallest_val=def_smallest_val
            ):
        
        self.f_objective = f_objective
        self.dimension = dimension
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.max_fes = max_fes
        self.checkpoints = checkpoints
        self.smallest_val = smallest_val
        self.objective_counter = 0

        self.pop_size = pop_size or self.dimension * 5
        self.memory_size = memory_size or self.dimension * 20
        self.archive_size = archive_size or self.dimension * self.pop_size
        
        #[populant][wymiar]
        self.population = [[random.uniform(self.min_clamp, self.max_clamp) for j in range(self.dimension)] for i in range(self.pop_size)]        

        self.Fitmass = [0] * self.pop_size

        self.final_best_sol = None
        self.final_best_fit = None

        self.memory_Cr = [0.2] * self.memory_size
        self.memory_F = [0.2] * self.memory_size
        self.memory_current_index = 0
        self.current_archive_size = 0

        if(X is None):
            self.X = [[random.uniform(self.min_clamp, self.max_clamp) for _ in range(self.dimension)] for _ in range(self.pop_size)]
        else:
            self.X = X
        
        if(objective_limit is None):
            self.objective_limit = self.dimension * self.max_fes
        else:
            self.objective_limit = objective_limit
        
        #ArchProbs
        self.arch_use_prob = 0.5
        self.arch_usages = [0] * self.pop_size

#TODO pamiętaj że f_objective zwraca już różnicę między optimum a aktualnym

    def check_point(self, index): # FindNSaveBest
        if(self.final_best_fit is None or self.Fitmass[index] < self.final_best_fit):
            self.final_best_fit = self.Fitmass[index]
            self.final_best_sol = self.X[index]


    def check_generated(self, num, rand_points_3, cur_indx):
        if(rand_points_3[num] == cur_indx):
            return False
        for j in range(num):
            if(rand_points_3[j] == rand_points_3[num]):
                return False
        return True

    # dodano sprawdzanie prohibited
    # GenerateNextRandUnif
    def generate_rand(self, num, rand_points_3, cur_indx):
        for j in range(RETRIES):
            generate_again = False
            rand_points_3[num] = random.randrange(self.pop_size)
            for i in range(num):
                if(rand_points_3[i] == rand_points_3[num] or rand_points_3[num] == cur_indx):
                    generate_again = True
            if(not generate_again):
               break

    # TODO -  archiwum to część pamięci dopisana na końcu populacji - nie lepiej osobno... tak normalnie po ludzku?
    def generate_rand_archive_only(self, num, rand_points_3, cur_indx):
        for j in range(RETRIES):
            generate_again = False
            rand_points_3[num] = random.randrange(self.current_archive_size) + self.pop_size
            for i in range(num):
                if(rand_points_3[i] == rand_points_3[num] or rand_points_3[num] == cur_indx):
                    generate_again = True
            if(not generate_again):
               break

    def main_cycle(self):
        mean_indiv = [0] * self.dimension
        mean_indiv_old = [0] * self.dimension
        popul_lim_count = [0] * self.pop_size
        num_of_stagIt=0
        rand_points_3 = [None, None, None]
        for cur_indx in range(self.pop_size):
            self.Fitmass[cur_indx] = self.f_objective(self.X[cur_indx])
            self.check_point(cur_indx)

        while(self.objective_counter < self.objective_limit):

            self.FitmassCopy = self.Fitmass
            Indexes = np.arange(len(FitmassCopy))

            if(np.max(self.Fitmass) != np.min(self.Fitmass)):
                sort_idx = np.argsort(FitmassCopy)
                FitmassCopy = FitmassCopy[sort_idx]
                Indexes = Indexes[sort_idx]

                BackIndexes = np.empty_like(Indexes)
                for j, idx in enumerate(Indexes):
                    BackIndexes[idx] = j
                
                # FitTemp3
                fit_temp3 = [0] * self.pop_size
                for i in range(self.pop_size):
                    fit_temp3[i] = math.exp(-i/self.pop_size)

                psizeval = max(2.0,self.pop_size*(0.2/self.max_fes*self.objective_counter+0.2))
                
                CrossExponential = 0
                if(random(0,1) < 0.5):
                    CrossExponential = 1

                generated_F = []
                generated_Cr = []
                for cur_indx in range(self.pop_size):
                    self.memory_current_index = random.randrange(self.memory_size)
                    Cr = min(1.0,max(0.0,random.uniform(self.memory_Cr[self.memory_current_index],0.1)))
                    while True:
                        F = self.memory_F[self.memory_current_index] + 0.1 * np.random.standard_cauchy()
                        if(F > 0):
                            break
                    generated_F.append(min(F,1.0))
                    generated_Cr.append(Cr)
            generated_Cr.sort()

            for cur_indx in range(self.pop_size):
                rand_points_3[0] = Indexes[random.randrange(psizeval)]
                for i in range(RETRIES):
                    if self.check_generated(0, rand_points_3, cur_indx):
                        break
                    rand_points_3[0] = Indexes[random.randrange(psizeval)]
                
                self.generate_rand(1, rand_points_3, cur_indx)

                if(random.uniform(0, 1) > self.arch_use_prob or self.current_archive_size == 0):
                    #ComponentSelector3
                    for i in range(RETRIES):
                        rand_points_3[2] = random.choices(range(self.pop_size), weights=fit_temp3)[0]
                        if self.check_generated(2, rand_points_3, cur_indx):
                            break
                    # TODO - necessary if starts as zeroes everywhere?
                    self.arch_usages[cur_indx] = 0
                else:
                    self.generate_rand_archive_only(2, rand_points_3, cur_indx)
                    self.arch_usages[cur_indx] = 1
                
                Donor = [] * self.dimension
                for j in range(self.dimension):
                    Donor[j] = (
                        self.population[cur_indx][j] + 
                        generated_F[cur_indx] * (self.population[rand_points_3[0]][j] - self.population[cur_indx][j]) + 
                        generated_F[cur_indx] * (self.population[rand_points_3[1]][j] - self.population[rand_points_3[2]][j])
                        )
                # zmiana po ustawieniu 
                F = generated_F[cur_indx]

# ComponentSelector3 brak
# Rands[2] = random.choices(range(popSize), weights=fit_temp3)[0]


#Rands -> rand_points_3
# DE style three parents indexes