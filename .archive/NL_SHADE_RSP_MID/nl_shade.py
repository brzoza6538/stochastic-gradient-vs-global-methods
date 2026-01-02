import random
import numpy as np
import math
import copy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from globals import *

CR_SIZE_PARAM = 0.5
CR_MIN = 0.0
MAX_F = 1.0
CAUCHY_F_SIZE = 0.1
F_INIT = 0.2
CR_INIT = 0.2

# TODO - uses lists - the rest algos use np.arrays
MIN_POP_SIZE = 2000  # działa dla 2000
ARCHIVE_SIZE_PARAM = 2.1
PSIZEVAL_RANGE = [0.2, 0.2] 
PSIZEVAL_MIN = 2.0
RETRIES = 25

RESAMPLING = True 
RESAMPLING_TRIALS_LIMIT = 100
NUM_OF_TRIALS_BEFORE_F_CHANGE = 10

COUNT_LIMITS = True ###################
MIN_ITERATIONS_ON_BOUND = 9

K_MEANS_AS_NEAREST = True #####################
MAX_K = 2
MIN_POP_SIZE_4_SPLIT = 4
MIN_DIST = 1e-5
MAX_NUM_OF_STAG_IT = 7

class NL_SHADE_RSP_MID():
    def __init__(
            self,
            f_objective,
            dimension,
            X=None,
            pop_size=None,
            memory_size=None,
            archive_size_param=None,
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
        self.seen_checkpoints = set()

        self.smallest_val = smallest_val
        self.objective_counter = 0

        self.log = {checkpoint: [] for checkpoint in self.checkpoints}

        self.pop_size = pop_size or self.dimension * 10
        self.memory_size = memory_size or self.dimension * 20

        self.archive_size_param = archive_size_param or ARCHIVE_SIZE_PARAM
        self.archive_size = self.dimension * math.ceil(self.archive_size_param)

        self.fitmass = [0] * self.pop_size
        self.max_pop_size = self.pop_size


        #bestSOL
        self.final_best_sol = None
        # globalBestFit
        self.final_best_fit = None

        self.iter_best_fit = None
        self.iter_best_sol = [None] * self.dimension

        #TODO - remove later
        self.succ_log = []
        self.AAA = 0 

        self.memory_Cr = [CR_INIT] * self.memory_size
        self.memory_F = [F_INIT] * self.memory_size
        self.current_archive_size = 0
        
        # self.population
        #[populant][wymiar]
        if(X is None):
            self.X = [[random.uniform(self.min_clamp, self.max_clamp) for _ in range(self.dimension)] for _ in range(self.pop_size)]
        else:
            self.X = X
        
        self.archive = [[None for j in range(self.dimension)] for i in range(self.max_pop_size*math.ceil(self.archive_size_param))]        

        if(objective_limit is None):
            self.objective_limit = self.dimension * self.max_fes
        else:
            self.objective_limit = objective_limit
        
        #ArchProbs
        self.arch_use_prob = 0.5
        self.arch_usages = [0] * self.pop_size
        #populLimCount
        self.popul_lim_count = [0] * self.pop_size
        # populTemp
        self.temp_pop = [0] * self.pop_size

        self.pop_fit_tmp = [0] * self.pop_size

        self.memory_iter = 0
        self.success_counter = 0

        self.temp_success_cr = []
        self.temp_success_f = []
        self.temp_success_fit_delta = []
        

        self.mean_indiv = [0] * self.dimension
        self.mean_indiv_old = [0] * self.dimension
        self.num_of_stag_it = 0

        self.generation = 0
        
    def save_success_cr_f(self, Cr, F, FitD):
        self.temp_success_cr.append(Cr)
        self.temp_success_f.append(F)
        self.temp_success_fit_delta.append(FitD)
        # SuccessFilled
        self.success_counter += 1

    def update_memory_cr_f(self):
        if self.success_counter != 0:
            self.memory_Cr[self.memory_iter] = self.mean_wl_general(self.temp_success_cr, self.temp_success_fit_delta, self.success_counter, 2, 1)
            self.memory_F[self.memory_iter] = self.mean_wl_general(self.temp_success_f, self.temp_success_fit_delta, self.success_counter, 2, 1)
            self.memory_iter += 1
            
            if self.memory_iter >= self.memory_size:
                self.memory_iter = 0
        else:
            self.memory_F[self.memory_iter] = 0.5
            self.memory_Cr[self.memory_iter] = 0.5

    def mean_wl_general(self, Vector, TempWeights, SuccessFilled, g_p, g_m):
        vec = Vector[:SuccessFilled]
        weights_raw = TempWeights[:SuccessFilled]
        
        sum_weight = np.sum(weights_raw)
        if sum_weight > 0:
            weights = weights_raw / sum_weight
        else:
            weights = np.ones_like(weights_raw) / len(weights_raw)  

        SumSquare = np.sum(weights * np.power(vec, g_p))
        Sum = np.sum(weights * np.power(vec, g_p - g_m))
        
        if abs(Sum) > 1e-6:
            return SumSquare / Sum
        else:
            return 0.5

    def get_val(self, index, j):
            if(index < self.pop_size):
                return self.X[index][j]
            else:
                return self.archive[index-self.pop_size][j]


#TODO pamiętaj że f_objective zwraca już różnicę między optimum a aktualnym

    def check_point(self, index): # FindNSaveBest
        if(self.iter_best_fit is None or self.fitmass[index] < self.iter_best_fit):
            self.iter_best_fit = self.fitmass[index]

            self.iter_best_sol = copy.deepcopy(self.X[index]) # TODO - po co iter_best_indx?

        if self.final_best_fit is None or self.iter_best_fit < self.final_best_fit:
            self.final_best_fit = self.iter_best_fit
            self.final_best_sol = copy.deepcopy(self.X[index])

    # not IsInfeasable 
    def check_if_in_range(self, index):
        for j in range(self.dimension):
            if self.temp_pop[index][j] < self.min_clamp or self.temp_pop[index][j] > self.max_clamp:
                return False
        return True

    # def count_broken_limits_streak(self, index):
    #     on_bound = False
    #     for j in range(self.dimension):
    #         if self.temp_pop[j] < self.min_clamp or self.temp_pop[j] > self.max_clamp:
    #             self.popul_lim_count[index] += 1
    #             return
            
    #     self.popul_lim_count = 0
    #     return 
                # FindLimits
    def fix_point_the_hard_way(self, individual):
        for j in range(self.dimension):
            if individual[j] < self.min_clamp or individual[j] > self.max_clamp:
                individual[j] = random.uniform(self.min_clamp, self.max_clamp)

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

    def generate_rand_archive_only(self, num, rand_points_3, cur_indx):
        for j in range(RETRIES):
            generate_again = False
            rand_points_3[num] = random.randrange(self.current_archive_size) + self.pop_size
            for i in range(num):
                if(rand_points_3[i] == rand_points_3[num] or rand_points_3[num] == cur_indx):
                    generate_again = True
            if(not generate_again):
               break

    def copy_to_archive(self, individual):
        # jeśli archiwum się zmniejszyło, a potem zwiększyło
        while len(self.archive) <= self.current_archive_size:
            self.archive.append(None)

        if self.current_archive_size < self.archive_size:
            self.archive[self.current_archive_size] = individual.copy()
            self.current_archive_size += 1

        elif self.archive_size > 0:
            to_go = random.randrange(self.archive_size)
            self.archive[to_go] = individual.copy()



    def collect_data(self): #SaveBestValues
        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * self.objective_limit)

            if self.final_best_fit < self.smallest_val and self.objective_counter <= checkpoint_fes:
                self.log[checkpoint].append(0)

            if checkpoint not in self.seen_checkpoints and self.objective_counter >= checkpoint_fes:
                self.log[checkpoint].append(0 if self.final_best_fit < self.smallest_val else (self.AAA, self.final_best_fit))
                self.seen_checkpoints.add(checkpoint)

    # RemoveWorst
    def adapt_pop_size(self, new_size):
        #TODO - RemoveWorst nie działało dla zwiększania populacji
        # - przywracało tylko pola które wcześniej nie używało ze starymi wartościami
        # - [a(val=2), b(val=5), c(val=1)] -> removeWorst(popsize-1) ->  [a, c, c]


        # points_to_remove = self.pop_size - new_size

        # for L in range(points_to_remove):
        #     worst_ind = 0
        #     worst_fit = self.fitmass[worst_ind]

        #     for i in range(1, self.pop_size):
        #         if self.fitmass[i] > worst_fit:
        #             worst_fit = self.fitmass[i]
        #             worst_ind = i
            
        #     for i in range(worst_ind, self.pop_size-1):
        #         for j in range(self.dimension):
        #             self.X[i][j] = self.X[i+1][j]
        #         self.fitmass[i] = self.fitmass[i+1]
        
        # kod który autentycznie przycina tablice
        best_indices = sorted(range(len(self.fitmass)), key=lambda i: self.fitmass[i])[:new_size]

        # przycinamy X i fitmass
        self.X = [self.X[i] for i in best_indices]
        self.fitmass = [self.fitmass[i] for i in best_indices]

        # ustaw pop_size na faktyczny rozmiar listy
        self.pop_size = best_indices



        

    def start(self):
        #populTemp
        self.temp_pop = [[None for j in range(self.dimension)] for i in range(self.pop_size)]        

        for cur_indx in range(self.pop_size):
            self.fitmass[cur_indx], evals_used = self.f_objective(self.X[cur_indx])

            self.objective_counter += evals_used
            self.check_point(cur_indx)

        end = False 
        loop = 0
        while(self.objective_counter < self.objective_limit) and (self.final_best_fit is None or self.final_best_fit > self.smallest_val) and not end:
            loop += 1
            print("loop : ", loop, "obj cntr: ", self.objective_counter, "poop_size: ", self.pop_size)
            end = self.step()
            self.collect_data()


    def step(self):   
        fitmass_copy = np.array(self.fitmass)
        indexes = np.arange(len(fitmass_copy))

        if(np.max(self.fitmass) != np.min(self.fitmass)):
            sort_idx = np.argsort(fitmass_copy)
            fitmass_copy = fitmass_copy[sort_idx]
            indexes = indexes[sort_idx]

            BackIndexes = np.empty_like(indexes)
            for j, idx in enumerate(indexes):
                BackIndexes[idx] = j
            
            # FitTemp3
            fit_temp3 = [0] * self.pop_size
            for i in range(self.pop_size):
                fit_temp3[i] = math.exp(-i/self.pop_size)

            psizeval = int(max(PSIZEVAL_MIN,self.pop_size*(PSIZEVAL_RANGE[0]/self.objective_limit*self.objective_counter+PSIZEVAL_RANGE[1])))



            cross_exponential = 0
            if(random.random() < 0.5):
                cross_exponential = 1

            generated_F = []
            generated_Cr = []
            # TODO - check in later next to TODO - second 235
            for cur_indx in range(self.pop_size):
                memory_current_index = random.randrange(self.memory_size)
                Cr = min(1.0,max(CR_MIN,random.uniform(self.memory_Cr[memory_current_index],CR_SIZE_PARAM)))
                while True:
                    F = self.memory_F[memory_current_index] + CAUCHY_F_SIZE * np.random.standard_cauchy()
                    if(F > 0):
                        break
                generated_F.append(min(F,MAX_F))
                generated_Cr.append(Cr)
                
            generated_Cr.sort()


# main-main loop
        for cur_indx in range(self.pop_size):
            rand_points_3 = [None, None, None]

            rand_points_3[0] = indexes[random.randrange(psizeval)]
            for i in range(RETRIES):
                if self.check_generated(0, rand_points_3, cur_indx):
                    break
                rand_points_3[0] = indexes[random.randrange(psizeval)]
            
            self.generate_rand(1, rand_points_3, cur_indx)

            if(random.random() > self.arch_use_prob or self.current_archive_size == 0):
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
            
            donor = [0] * self.dimension



            for j in range(self.dimension):

                donor[j] = (
                    self.X[cur_indx][j] + 
                    generated_F[cur_indx] * (self.get_val(rand_points_3[0], j) - self.get_val(cur_indx,j)) + 
                    generated_F[cur_indx] * (self.get_val(rand_points_3[1], j) - self.get_val(rand_points_3[2],j))
                    )
            # zmiana po ustawieniu 

            # self.fix_point_the_hard_way(donor)
            F = generated_F[cur_indx]
            Cr = generated_Cr[BackIndexes[cur_indx]]

            will_crossover = random.randrange(self.dimension)
            
            # zwiększ krzyżowanie na więcej niż jedno tylko w drugiej połowie
            Cr_to_use = 0
            if self.objective_counter > (0.5 * self.objective_limit):
                Cr_to_use = (self.objective_counter/self.objective_limit - 0.5) * 2

            if cross_exponential == 0:
                for j in range(self.dimension):
                    if random.random() < Cr_to_use or will_crossover == j:
                        self.temp_pop[cur_indx][j] = donor[j]
                    else: 
                        self.temp_pop[cur_indx][j] = self.X[cur_indx][j]
            else:
                start_loc = random.randrange(self.dimension)
                L = start_loc + 1

                while random.random() < Cr and L < self.dimension:
                    L += 1
                for j in range(self.dimension):
                        self.temp_pop[cur_indx][j] = self.X[cur_indx][j]

                for j in range(start_loc, L):
                    self.temp_pop[cur_indx][j] = donor[j]
            


##################### REASAMPLING
            if RESAMPLING:
                used_repair = False
                num_of_trials = 1

                while(not self.check_if_in_range(cur_indx) and num_of_trials<=RESAMPLING_TRIALS_LIMIT):
                    used_repair = True
                    if num_of_trials>NUM_OF_TRIALS_BEFORE_F_CHANGE:
                        # TODO - second 235
                        cross_exponential = 0
                        if(random.random() < 0.5):
                            cross_exponential = 1
                        
                        memory_current_index = random.randrange(self.memory_size)

                        Cr = min(1.0,max(CR_MIN,random.uniform(self.memory_Cr[memory_current_index],CR_SIZE_PARAM)))
                        while True:
                            F = self.memory_F[memory_current_index] + CAUCHY_F_SIZE * np.random.standard_cauchy()
                            if(F > 0):
                                break
                        generated_F[cur_indx] = min(F,MAX_F)
                        generated_Cr[cur_indx] = Cr
                    
                    # TODO duplication - make into a func?
                    rand_points_3[0] = indexes[random.randrange(psizeval)]

                    for i in range(RETRIES):
                        if self.check_generated(0, rand_points_3, cur_indx):
                            break
                        rand_points_3[0] = indexes[random.randrange(psizeval)]
                    
                    self.generate_rand(1, rand_points_3, cur_indx)

                    if(random.random() > self.arch_use_prob or self.current_archive_size == 0):
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

                    donor = [0] * self.dimension
                    for j in range(self.dimension):
                        donor[j] = (
                            self.X[cur_indx][j] + 
                            generated_F[cur_indx] * (self.get_val(rand_points_3[0], j) - self.X[cur_indx][j]) + 
                            generated_F[cur_indx] * (self.get_val(rand_points_3[1], j) - self.get_val(rand_points_3[2], j))
                            )
                    # TODO duplication - make into a func? ^
                    # self.fix_point_the_hard_way(donor)

                    F = generated_F[cur_indx]
                    Cr = generated_Cr[BackIndexes[cur_indx]]
                    will_crossover = random.randrange(self.dimension)
                    Cr_to_use = 0

                    if self.objective_counter > (0.5 * self.objective_limit):
                        Cr_to_use = (self.objective_counter/self.objective_limit - 0.5) * 2

                    if cross_exponential == 0:
                        for j in range(self.dimension):
                            if random.random() < Cr_to_use or will_crossover == j:
                                self.temp_pop[cur_indx][j] = donor[j]
                            else: 
                                self.temp_pop[cur_indx][j] = self.X[cur_indx][j]
                    else:
                        start_loc = random.randrange(self.dimension)
                        L = start_loc + 1

                        while random.random() < Cr and L < self.dimension:
                            L += 1

                        for j in range(self.dimension):
                                self.temp_pop[cur_indx][j] = self.X[cur_indx][j]

                        for j in range(start_loc, L):
                            self.temp_pop[cur_indx][j] = donor[j]
                    
                    num_of_trials += 1

                    if(not self.check_if_in_range(cur_indx)):
                        used_repair = False
                        self.fix_point_the_hard_way(self.temp_pop[cur_indx])


##################### ^RESAMPLING end 

##################### COUNT LIMITS  - part of RESAMPLING
                if COUNT_LIMITS:
                    if self.popul_lim_count[cur_indx]>MIN_ITERATIONS_ON_BOUND:
                        print("\COUNT_LIMITS\n")
                        return True
##################### ^COUNT LIMITS end 

            # TODO - pamiętaj od razu zwraca error - żeby nie trzeba było podawać tu optFit
            self.pop_fit_tmp[cur_indx], evals_used = self.f_objective(self.temp_pop[cur_indx])
            self.objective_counter += evals_used

            if self.iter_best_fit is None or (self.pop_fit_tmp[cur_indx] < self.iter_best_fit):
                self.iter_best_fit = self.pop_fit_tmp[cur_indx]
                self.iter_best_sol = copy.deepcopy(self.X[cur_indx])
            
                if self.final_best_fit is None or (self.pop_fit_tmp[cur_indx] < self.final_best_fit):
                    self.iter_best_fit = self.pop_fit_tmp[cur_indx]
                    self.final_best_sol = copy.deepcopy(self.temp_pop[cur_indx])

                    # if self.pop_fit_tmp[cur_indx] <= self.smallest_val:
                    #     return(self.objective_counter, self.final_best_sol)

            if self.pop_fit_tmp[cur_indx] < self.fitmass[cur_indx]:
                self.save_success_cr_f(Cr, F, self.fitmass[cur_indx] - self.pop_fit_tmp[cur_indx])

            self.check_point(cur_indx)
            # if(self.objective_counter > self.max_fes):
            #     return(self.objective_counter, self.final_best_sol)
##################### K_MEANS_AS_NEAREST 


        if K_MEANS_AS_NEAREST:  
            
            min_sil_score = 1/(4*math.sqrt(self.dimension))

            best_silhouette = None 
            best_k = None
            best_assignments = None
            best_centroids = None

            data = copy.deepcopy(self.temp_pop)
            for cand_k in range(2, MAX_K + 1):
                kmeans = KMeans(n_clusters=cand_k, n_init=10)

                assignments = kmeans.fit_predict(data)
                centroids = kmeans.cluster_centers_

                silhouette = silhouette_score(data, assignments, metric='euclidean')

                if best_silhouette is None or silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = cand_k
                    best_assignments = assignments
                    best_centroids = centroids
            

            bestCandFit = None
            
            if best_silhouette > min_sil_score and self.pop_size >= MIN_POP_SIZE_4_SPLIT:
                for k in range(best_k):
                    for i in range(self.dimension):
                        self.mean_indiv = best_centroids[k]
                        self.fix_point_the_hard_way(self.mean_indiv)

                    fit_mean, evals_used = self.f_objective(self.mean_indiv)
                    self.objective_counter += evals_used
                    
                    if bestCandFit is None or fit_mean < bestCandFit:
                        bestCandFit = fit_mean
                    

                    if self.final_best_fit is None or fit_mean < self.final_best_fit:
                        self.final_best_fit = fit_mean
                        self.final_best_sol = copy.deepcopy(self.mean_indiv)
                    
            else:
                #if kmean works too bad
                self.mean_indiv = np.mean(self.temp_pop, axis=0)
                self.fix_point_the_hard_way(self.mean_indiv)
                
                fit_mean, evals_used = self.f_objective(self.mean_indiv)
                self.objective_counter += evals_used

                if self.final_best_fit is None or fit_mean < self.final_best_fit:
                    self.final_best_fit = fit_mean
                    self.final_best_sol = copy.deepcopy(self.mean_indiv)

            # switch closest point to it, into mean
            # TODO - why in resampling?

            # chosen_indx = np.argmin(np.linalg.norm(self.temp_pop - self.mean_indiv, axis=1))
            # if fit_mean < self.pop_fit_tmp[chosen_indx]:
            #     self.pop_fit_tmp[chosen_indx] = fit_mean
            #     self.temp_pop[chosen_indx] = self.mean_indiv.copy()


            self.mean_indiv = np.mean(self.temp_pop, axis=0)
            dist = np.linalg.norm(self.mean_indiv_old - self.mean_indiv)
            self.AAA = (fit_mean, dist)

            if dist < MIN_DIST:
                self.num_of_stag_it += 1
                if self.num_of_stag_it > MAX_NUM_OF_STAG_IT:
                    print("\nSTAGNANT\n")
                    return True
            else:
                self.num_of_stag_it = 0

            self.mean_indiv_old = self.mean_indiv
##################### ^ K_MEANS_AS_NEAREST  end


            # ArchSuccess
        archiv_succ = 0
        no_archiv_succ = 0
        arch_usages_counter = 0
        
        for cur_indx in range(self.pop_size):


            if(self.pop_fit_tmp[cur_indx] < self.fitmass[cur_indx]):
                if self.arch_usages[cur_indx] == 1: 
                    archiv_succ += (self.fitmass[cur_indx] - self.pop_fit_tmp[cur_indx]) / self.fitmass[cur_indx]
                    arch_usages_counter += 1
                else:
                    no_archiv_succ += (self.fitmass[cur_indx] - self.pop_fit_tmp[cur_indx]) / self.fitmass[cur_indx]
                
                self.copy_to_archive(self.X[cur_indx])
            
            # TODO - was this to avoid troubles if previous if changes pop_fit_tmp or just a mistake?
            # AAAA pop change
            if(self.pop_fit_tmp[cur_indx] <= self.fitmass[cur_indx]):
                self.X[cur_indx] = self.temp_pop[cur_indx]
                self.fitmass[cur_indx] = self.pop_fit_tmp[cur_indx]
        
        if arch_usages_counter != 0:
            archiv_succ = archiv_succ/arch_usages_counter
                
            no_archiv_succ = no_archiv_succ / (self.pop_size - arch_usages_counter)
            
            self.arch_use_prob = archiv_succ / (archiv_succ + no_archiv_succ)
            self.arch_use_prob = max(0.1, min(0.9, self.arch_use_prob))
            if archiv_succ == 0:
                self.arch_use_prob = 0.5
        else:
            self.arch_use_prob = 0.5



        # TODO - dodaj isRestart
        # if(isRestart){//HomoAtReset
        #     double shapeConst = 0.1;//jak większe to szybszy spadek, mozna tez pokręcić minimalnym rozmiarem pop
        #     //double shapeConst = 0.2;//jak większe to szybszy spadek, mozna tez pokręcić minimalnym rozmiarem pop
        #     //double shapeConst = 0.05;

        #     //const int MIN_POP=4;

        #     const int MIN_POP=20;//best

        #     //const int MIN_POP=30;
        #     //const int MIN_POP=10;

        #     double divider = (maxFES-evalsAtStart)/shapeConst;

        #     double delta=pow( (MIN_POP* (maxFES-evalsAtStart)/divider-(maxFES-evalsAtStart)/divider*popSize), 2)-4*(maxFES-evalsAtStart)/divider*(MIN_POP-popSize);
        #     if(delta<=0){
        #         newNInds=MIN_POP;
        #     }else{
        #     double b1=(-(MIN_POP*(maxFES-evalsAtStart)/divider-(maxFES-evalsAtStart)/divider*popSize)-sqrt(delta))/(2*(MIN_POP-popSize));
        #     double a1=popSize-1/b1;
        #     newNInds = round(a1+1/( (NFEval-evalsAtStart) /divider+b1));
        #     }
        # }else{
        #     newNInds = round((NIndsMin-NIndsMax)*pow((double(NFEval)/double(maxFES)),(1.0-double(NFEval)/double(maxFES)))+NIndsMax);
        # }


        newNInds = round((MIN_POP_SIZE-self.max_pop_size)*pow((self.objective_counter/self.objective_limit),(1.0-self.objective_counter/self.objective_limit))+self.max_pop_size)
        
        if(newNInds < MIN_POP_SIZE):
            newNInds = MIN_POP_SIZE
        if(newNInds > self.max_pop_size):
            newNInds = self.max_pop_size

        new_arch_size = round(((MIN_POP_SIZE-self.max_pop_size)*pow((self.objective_counter/self.objective_limit),(1.0-self.objective_counter/self.objective_limit))+(self.max_pop_size * self.archive_size_param)))
        
        if(new_arch_size < MIN_POP_SIZE):
            new_arch_size = MIN_POP_SIZE
        if self.current_archive_size >= new_arch_size:
            self.archive = self.archive[:new_arch_size]
            self.current_archive_size = new_arch_size

        self.archive_size = new_arch_size
        self.adapt_pop_size(newNInds)

        self.pop_size = newNInds

        self.update_memory_cr_f()

        # TODO - remove later
        self.succ_log.append(self.success_counter)
        self.success_counter = 0
        self.generation += 1

        return False
            
# TODO - pamiętaj że collect data zmieniłeś żeby robiło na końcu każdej pętli
# TODO - pamiętaj że sprawdza czy najlepszy wynik jest mniejszy od smallest val i wychodzi dopiero pod koniec pętli a nie w trakcie jak wcześniej
# max_fes to 10 000 - ile może zmienić dokończenie pętli z populacją dimension * 5
# TODO - check every var you init with none
