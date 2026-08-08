import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from algorithms import *

from sklearn.exceptions import ConvergenceWarning
import warnings

F_INIT = 0.2
CR_INIT = 0.2
RETRIES = 25
RESAMPLING = True
MAX_NUM_OF_TRIALS = 20
NUM_OF_TR_WITHOUT_F_HANGE = 10

COUNT_LIMITS = True
MIN_ITS_ON_BOUND = 9

K_MEANS_AS_NEAREST = True 
MAX_K = 2
MIN_POP_SIZE_4_SPLIT = 4
MIN_DIST = 1e-8
MAX_NUM_OF_STAG_IT = 7

class NL_SHADE_RSP_MID():
    def __init__(
            self,
            f_objective,
            dimension,
            X=None,
            max_fes=def_max_fes,
            objective_limit=None,
            min_clamp=def_clamps[0],
            max_clamp=def_clamps[1],
            checkpoints=def_checkpoints,
            smallest_val=def_smallest_val,
            pop_size=None,
            memory_size=None
            ):
        
        self.max_fes = max_fes
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.checkpoints = checkpoints
        self.smallest_val = smallest_val
        self.f_objective = f_objective
        self.dimension = dimension

        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.help_log = {checkpoint: [] for checkpoint in self.checkpoints}

        self.checkpoints = checkpoints
        self.seen_checkpoints = set()
        self.objective_limit = objective_limit or self.dimension * self.max_fes
        self.objective_counter = 0

# TODO really * 5?
        self.pop_size = pop_size or self.dimension * 5
        self.min_pop_size = 4
        self.max_pop_size = self.pop_size

        self.arch_size_param = 2.1
        self.memory_size = memory_size or self.dimension * 20
        self.memory_iter = 0
        self.arch_size = self.pop_size * math.ceil(self.arch_size_param)
        self.curr_arch_size = 0

        if(X is None):
            self.pop = np.array([[random.uniform(self.min_clamp, self.max_clamp) for _ in range(self.dimension)] for _ in range(self.pop_size)])
        else:
            self.pop = X
        self.archive = np.array([[None for _ in range(self.dimension)] for _ in range(self.arch_size)])

        self.fitmass = np.zeros(self.pop_size)

        self.Cr_memory = np.ones(self.memory_size) * CR_INIT
        self.F_memory = np.ones(self.memory_size) * F_INIT

        self.indexes = np.array([None for _ in range(self.pop_size)])
        self.backindexes = np.array([None for _ in range(self.pop_size)])

        self.arch_usages = np.zeros(self.pop_size)
        self.pop_lim_count = np.zeros(self.pop_size)
        self.arch_use_prob = 0.5

        self.global_best_fit = None
        self.global_best_sol = None

        self.temp_succ_Cr = []
        self.temp_succ_F = []
        self.fit_delta = []

        self.mean_indiv = np.zeros(self.dimension)
        self.mean_indiv_old = np.zeros(self.dimension)


    def mean_wl_general(self,vector, temp_weights, g_p=2, g_m=1):
        vec = np.array(vector, dtype=float)
        weights_raw = np.array(temp_weights, dtype=float)
        
        sum_weight = np.sum(weights_raw)
        if sum_weight > 0:
            weights = weights_raw / sum_weight
        else:
            weights = np.ones_like(weights_raw) / len(weights_raw)
        
        sum_square = np.sum(weights * np.power(vec, g_p))
        sum = np.sum(weights * np.power(vec, g_p - g_m))
        
        if abs(sum) > 1e-6:
            return sum_square / sum
        else:
            return 0.5

    def update_memory_CrF(self):
        if(len(self.fit_delta) != 0):
            self.Cr_memory[self.memory_iter] = self.mean_wl_general(self.temp_succ_Cr, self.fit_delta, 2, 1)
            self.F_memory[self.memory_iter] = self.mean_wl_general(self.temp_succ_F, self.fit_delta, 2, 1)
        
        else:
            self.Cr_memory[self.memory_iter] = 0.5
            self.F_memory[self.memory_iter] = 0.5

        self.memory_iter += 1
        if(self.memory_iter >= self.memory_size):
            self.memory_iter = 0


        self.temp_succ_Cr = []
        self.temp_succ_F = []
        self.fit_delta = []





    def collect_data(self, end=False): # TODO error dodawany jako lista

        if end:
            for checkpoint in self.checkpoints:
                if checkpoint not in self.seen_checkpoints:
                    self.log[checkpoint].append(float(0 if self.global_best_fit < self.smallest_val else self.global_best_fit))
                    self.help_log[checkpoint].append(self.global_best_sol.copy())

                    self.seen_checkpoints.add(checkpoint)

        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * self.objective_limit)
            if self.global_best_fit < self.smallest_val and self.objective_counter <= checkpoint_fes:
                # print("--------------------")
                # print(f"chkpnt{checkpoint}  Cr : ", self.Cr_memory)
                # print("--------------------")
                # print(f"chkpnt{checkpoint}  F : ", self.F_memory)
                # print("--------------------")
                print(f"chpt {checkpoint} cntr: ", self.objective_counter)

                self.log[checkpoint].append(0)
            if checkpoint not in self.seen_checkpoints and self.objective_counter >= checkpoint_fes:
                # print("--------------------")
                # print(f"chkpnt{checkpoint}  Cr : ", self.Cr_memory)
                # print("--------------------")
                # print(f"chkpnt{checkpoint}  F : ", self.F_memory)
                # print("--------------------")
                # print(f"chpt {checkpoint} cntr: ", self.objective_counter)

                self.log[checkpoint].append(float(0 if self.global_best_fit < self.smallest_val else self.global_best_fit))
                self.help_log[checkpoint].append(self.global_best_sol.copy())

                self.seen_checkpoints.add(checkpoint)
        
            


    def resize_pop(self, new_pop_size):
        if new_pop_size < self.pop_size:
            best_indices = np.argpartition(
                self.fitmass,
                new_pop_size - 1
            )[:new_pop_size]

            self.pop = self.pop[best_indices].copy()
            self.fitmass = self.fitmass[best_indices].copy()
            self.pop_lim_count = self.pop_lim_count[best_indices].copy()

        elif new_pop_size > self.pop_size:

            points_to_add = new_pop_size - self.pop_size

            indices = np.random.randint(
                0,
                self.pop_size,
                size=points_to_add
            )

            self.pop = np.vstack((
                self.pop,
                self.pop[indices]
            ))

            self.fitmass = np.concatenate((
                self.fitmass,
                self.fitmass[indices]
            ))

            self.pop_lim_count = np.concatenate((
                self.pop_lim_count,
                self.pop_lim_count[indices]
            ))

        self.pop_size = new_pop_size

        self.arch_usages = np.zeros(self.pop_size)
        self.indexes = np.zeros(self.pop_size, dtype=int)
        self.backindexes = np.zeros(self.pop_size, dtype=int)

    def copy_to_archive(self, parent):
        if self.curr_arch_size < self.arch_size:
            self.archive[self.curr_arch_size] = copy.deepcopy(parent)
            self.curr_arch_size += 1

        elif self.arch_size > 0:
            self.archive[random.randrange(self.curr_arch_size)] = copy.deepcopy(parent)
                       

    def check_point(self, index): # FindNSaveBest
        if(self.global_best_fit is None or self.fitmass[index] < self.global_best_fit):
            self.global_best_fit = self.fitmass[index]
            self.global_best_sol = copy.deepcopy(self.pop[index])

    def check_parents(self, indx, parents, curr_indx):
        if(parents[indx] == curr_indx):
            return False
        for j in range(indx):
            if(parents[j] == parents[indx]):
                return False
        return True

    def generate_rand(self, num, parents, curr_indx):
        for j in range(RETRIES):
            generate_again = False
            parents[num] = random.randrange(self.pop_size)
            for i in range(num):
                if(parents[i] == parents[num] or parents[num] == curr_indx):
                    generate_again = True
            if(not generate_again):
               break


    def pick_parents(self, curr_indx):
        psizeval = int(max(2.0,self.pop_size*(0.2/self.objective_limit*self.objective_counter+0.2)))

        parents = [None, None, None]
        parents[0] = self.indexes[random.randrange(psizeval)]
        for i in range(RETRIES):
            if self.check_parents(0, parents, curr_indx):
                break
            parents[0] = self.indexes[random.randrange(psizeval)]
        
        self.generate_rand(1, parents, curr_indx)

        if(random.random() > self.arch_use_prob or self.curr_arch_size == 0):
            #ComponentSelector3
            for i in range(RETRIES):

                weights = 1 / np.array(self.fitmass)
                prob = weights / np.sum(weights)
                parents[2] = np.random.choice(len(self.fitmass), p=prob)

                if self.check_parents(2, parents, curr_indx):
                    break
            self.arch_usages[curr_indx] = 0
        else:
            parents[2] = random.randrange(self.curr_arch_size)
            self.arch_usages[curr_indx] = 1
        return parents



    def start(self):
        for curr_indx in range(self.pop_size):
            self.fitmass[curr_indx], evals_used = self.f_objective(self.pop[curr_indx])

            self.objective_counter += evals_used
            self.check_point(curr_indx)

        end = False 
        loop = 0
        while(self.objective_counter < self.objective_limit) and (self.global_best_fit is None or self.global_best_fit > self.smallest_val) and not end:
            loop += 1
            end = self.step()
            self.collect_data()
        self.collect_data(end=True)
            # if loop % 20 == 0:
            #     print("--------------------")
            #     print(f"loop {loop}  Cr : ", self.Cr_memory)
            #     print("--------------------")
            #     print(f"loop {loop}  F : ", self.F_memory)
            #     print("--------------------")

    def step(self):

        self.indexes = np.argsort(self.fitmass)
        pop_tmp = np.zeros_like(self.pop) 
        fit_tmp = np.zeros_like(self.fitmass)

        for j, idx in enumerate(self.indexes):
            self.backindexes[idx] = j


        cross_exponential = False
        if(random.random() < 0.5):
            cross_exponential = True

        F_generated = np.array([None for _ in range(self.pop_size)])
        Cr_generated = np.array([None for _ in range(self.pop_size)])


        for curr_indx in range(self.pop_size):
            curr_memory_indx = random.randrange(self.memory_size)

            Cr = min(1.0, max(0.0, random.normalvariate(self.Cr_memory[curr_memory_indx],0.1)))
            while True:
                F =  min(1.0, np.random.standard_cauchy() * self.F_memory[curr_memory_indx] + 0.1) 
                if(F > 0):
                    break

            F_generated[curr_indx] = F
            Cr_generated[curr_indx] = Cr
        Cr_generated = np.flipud(np.sort(Cr_generated))

        for curr_indx in range(self.pop_size):
            parents = self.pick_parents(curr_indx)
                    

            if self.arch_usages[curr_indx] == 0:
                donor = (
                    self.pop[curr_indx] + 
                    F_generated[curr_indx] * (self.pop[parents[0]] - self.pop[curr_indx]) + 
                    F_generated[curr_indx] * (self.pop[parents[1]] - self.pop[parents[2]])
                    )
            else:
                donor = (
                    self.pop[curr_indx] + 
                    F_generated[curr_indx] * (self.pop[parents[0]] - self.pop[curr_indx]) + 
                    F_generated[curr_indx] * (self.pop[parents[1]] - self.archive[parents[2]])
                    )
            
            F = F_generated[curr_indx]
            dim_to_crossover = random.randrange(self.dimension)
            Cr = Cr_generated[self.backindexes[curr_indx]]
            # HERE
            Cr_to_use = 0
            if self.objective_counter > (0.5 * self.objective_limit):
                Cr_to_use = (self.objective_counter/self.objective_limit - 0.5) * 2

            if cross_exponential == False:
                for j in range(self.dimension):
                    if random.random() < Cr_to_use or dim_to_crossover == j:
                        pop_tmp[curr_indx][j] = donor[j]
                    else:
                        pop_tmp[curr_indx][j] = self.pop[curr_indx][j]
                
            else:
                start_loc = random.randrange(self.dimension)
                L = start_loc + 1

                while random.random() < Cr and L < self.dimension:
                    L += 1

                pop_tmp[curr_indx] = np.concatenate((self.pop[curr_indx][:start_loc], donor[start_loc:L], self.pop[curr_indx][L:]))
                    
# ##################### RESAMPLING
            if RESAMPLING:
                num_of_trials=1
                used_repair=False
                in_range = np.all((pop_tmp[curr_indx] >= self.min_clamp) & (pop_tmp[curr_indx] <= self.max_clamp))
                
                while (not in_range and num_of_trials<=MAX_NUM_OF_TRIALS):
                    used_repair = True

                    # in_range = np.all((pop_tmp[curr_indx] >= self.min_clamp) & (pop_tmp[curr_indx] <= self.max_clamp))
                    if(num_of_trials > NUM_OF_TR_WITHOUT_F_HANGE):
                        cross_exponential = False
                        if(random.random() < 0.5):
                            cross_exponential = True

                        curr_memory_indx = random.randrange(self.memory_size)

                        Cr = min(1.0, max(0.0, random.normalvariate(self.Cr_memory[curr_memory_indx],0.1)))
                        while True:
                            F = min(1.0, np.random.standard_cauchy() * self.F_memory[curr_memory_indx] + 0.1)
                            if(F > 0):
                                break

                        F_generated[curr_indx] = F
                        Cr_generated[curr_indx] = Cr

                    parents = self.pick_parents(curr_indx)
                                


                    if self.arch_usages[curr_indx] == 0:
                        donor = (
                            self.pop[curr_indx] + 
                            F_generated[curr_indx] * (self.pop[parents[0]] - self.pop[curr_indx]) + 
                            F_generated[curr_indx] * (self.pop[parents[1]] - self.pop[parents[2]])
                            )
                    else:
                        donor = (
                            self.pop[curr_indx] + 
                            F_generated[curr_indx] * (self.pop[parents[0]] - self.pop[curr_indx]) + 
                            F_generated[curr_indx] * (self.pop[parents[1]] - self.archive[parents[2]])
                            )
                                
                    F = F_generated[curr_indx]
                    dim_to_crossover = random.randrange(self.dimension)

                    if(num_of_trials<=NUM_OF_TR_WITHOUT_F_HANGE):
                        Cr = Cr_generated[self.backindexes[curr_indx]]

                    Cr_to_use = 0
                    if self.objective_counter > (0.5 * self.objective_limit):
                        Cr_to_use = (self.objective_counter/self.objective_limit - 0.5) * 2

                    if cross_exponential == False:
                        for j in range(self.dimension):
                            if random.random() < Cr_to_use or dim_to_crossover == j:
                                pop_tmp[curr_indx][j] = donor[j]
                            else:
                                pop_tmp[curr_indx][j] = self.pop[curr_indx][j]
                            
                    else:
                        start_loc = random.randrange(self.dimension)
                        L = start_loc + 1

                        while random.random() < Cr and L < self.dimension:
                            L += 1

                        pop_tmp[curr_indx] = np.concatenate((self.pop[curr_indx][:start_loc], donor[start_loc:L], self.pop[curr_indx][L:]))

                    in_range = np.all((pop_tmp[curr_indx] >= self.min_clamp) & (pop_tmp[curr_indx] <= self.max_clamp))
                    if not in_range:
                        used_repair = False
                        for j in range(self.dimension):
                            if self.min_clamp > pop_tmp[curr_indx][j] or self.max_clamp < pop_tmp[curr_indx][j]:
                                pop_tmp[curr_indx][j] = np.random.uniform(self.min_clamp, self.max_clamp)

                    num_of_trials += 1



# ##################### ^RESAMPLING end 
                if COUNT_LIMITS:
                    in_range = np.all((pop_tmp[curr_indx] >= self.min_clamp) & (pop_tmp[curr_indx] <= self.max_clamp))
                    if not in_range:
                        self.pop_lim_count[curr_indx] += 1
                    else:
                        self.pop_lim_count[curr_indx] = 0
                    
                    # HERE
                    if self.pop_lim_count[curr_indx] > MIN_ITS_ON_BOUND:
                        self.collect_data(end=True)
                        return



# ##################### COUNT LIMITS  - part of RESAMPLING

# ##################### ^COUNT LIMITS end 
            fit_tmp[curr_indx], evals_used = self.f_objective(pop_tmp[curr_indx])
            self.objective_counter += evals_used

            if(self.global_best_fit is None or fit_tmp[curr_indx] < self.global_best_fit):
                self.global_best_fit = fit_tmp[curr_indx]
                self.global_best_sol = copy.deepcopy(pop_tmp[curr_indx])

            if(fit_tmp[curr_indx] < self.fitmass[curr_indx]):
                self.temp_succ_Cr.append(Cr)
                self.temp_succ_F.append(F)
                self.fit_delta.append(abs(fit_tmp[curr_indx] - self.fitmass[curr_indx]))

            self.check_point(curr_indx)


##################### K_MEANS_AS_NEAREST 

        if K_MEANS_AS_NEAREST:  
            
            min_sil_score = 1/(4*math.sqrt(self.dimension))

            best_silhouette = None 
            best_k = None
            best_assignments = None
            best_centroids = None
            bestCandFit = None
            
            data = copy.deepcopy(pop_tmp)
            for cand_k in range(2, MAX_K + 1):
                kmeans = KMeans(n_clusters=cand_k, n_init=10)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    assignments = kmeans.fit_predict(data)
                centroids = kmeans.cluster_centers_

                # Sprawdzenie liczby unikalnych klastrów
                if len(np.unique(assignments)) < 2:
                    # nie da się policzyć silhouette, pomijamy
                    continue

                silhouette = silhouette_score(data, assignments, metric='euclidean')

                if best_silhouette is None or silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = cand_k
                    best_assignments = assignments
                    best_centroids = centroids

            # Dalej: jeśli best_silhouette jest None (żadne sensowne klastry nie powstały)
            if best_silhouette is not None and best_silhouette > min_sil_score and self.pop_size >= MIN_POP_SIZE_4_SPLIT:
                for k in range(best_k):
                    self.mean_indiv = best_centroids[k].copy()
                    for j in range(self.dimension):
                        if self.min_clamp > self.mean_indiv[j] or self.max_clamp < self.mean_indiv[j]:
                            self.mean_indiv[j] = random.uniform(self.min_clamp, self.max_clamp)
                    fit_mean, evals_used = self.f_objective(self.mean_indiv)
                    self.objective_counter += evals_used
                    if bestCandFit is None or fit_mean < bestCandFit:
                        bestCandFit = fit_mean
                    if self.global_best_fit is None or fit_mean < self.global_best_fit:
                        self.global_best_fit = fit_mean
                        self.global_best_sol = copy.deepcopy(self.mean_indiv)
            else:
                # fallback: średnia populacji
                self.mean_indiv = np.mean(pop_tmp, axis=0)
                for j in range(self.dimension):
                    if self.min_clamp > self.mean_indiv[j] or self.max_clamp < self.mean_indiv[j]:
                        self.mean_indiv[j] = random.uniform(self.min_clamp, self.max_clamp)
                fit_mean, evals_used = self.f_objective(self.mean_indiv)
                self.objective_counter += evals_used
                if self.global_best_fit is None or fit_mean < self.global_best_fit:
                    self.global_best_fit = fit_mean
                    self.global_best_sol = copy.deepcopy(self.mean_indiv)

            chosen_indx = np.argmin(np.linalg.norm(pop_tmp - self.mean_indiv, axis=1))
            if fit_mean < fit_tmp[chosen_indx]:
                fit_tmp[chosen_indx] = fit_mean
                pop_tmp[chosen_indx] = self.mean_indiv.copy()


            self.mean_indiv = np.mean(pop_tmp, axis=0)
            dist = np.linalg.norm(self.mean_indiv_old - self.mean_indiv)

            if dist < MIN_DIST:
                self.num_of_stag_it += 1
                if self.num_of_stag_it > MAX_NUM_OF_STAG_IT:
                    print("\nSTAGNANT")
                    self.collect_data(end=True)
                    return True
            else:
                self.num_of_stag_it = 0

            self.mean_indiv_old = self.mean_indiv

##################### ^ K_MEANS_AS_NEAREST  end
        
        arch_succ = 0
        no_arch_succ = 0
        arch_use_cntr = 0

        for curr_indx in range(self.pop_size):
            if(fit_tmp[curr_indx] < self.fitmass[curr_indx]):
                if self.arch_usages[curr_indx] == 1:
                    arch_succ += (self.fitmass[curr_indx] - fit_tmp[curr_indx]) /self.fitmass[curr_indx]
                    arch_use_cntr += 1
                else:
                    no_arch_succ += (self.fitmass[curr_indx] - fit_tmp[curr_indx]) /self.fitmass[curr_indx]
                self.copy_to_archive(self.pop[curr_indx])

                self.pop[curr_indx] = pop_tmp[curr_indx]
                self.fitmass[curr_indx] = fit_tmp[curr_indx]
        
        if arch_use_cntr > 0:
            arch_succ = arch_succ / arch_use_cntr
        else:
            arch_succ = 0

        no_arch_count = self.pop_size - arch_use_cntr
        if no_arch_count > 0:
            no_arch_succ = no_arch_succ / no_arch_count
        else:
            no_arch_succ = 0

        # Update archive usage probability
        if arch_succ == 0:
            self.arch_use_prob = 0.5
        else:
            self.arch_use_prob = max(0.1, min(0.9, self.arch_use_prob))



        new_pop_size = round((self.min_pop_size-self.max_pop_size)*pow((self.objective_counter/self.objective_limit),(1.0-self.objective_counter/self.objective_limit))+self.max_pop_size)
        new_pop_size = int(max(self.min_pop_size, min(self.max_pop_size, new_pop_size)))

        new_arch_size = round((self.min_pop_size-self.max_pop_size)*pow((self.objective_counter/self.objective_limit),(1.0-self.objective_counter/self.objective_limit))+self.max_pop_size)*self.arch_size_param
        new_arch_size = int(max(self.min_pop_size, min(self.max_pop_size, new_arch_size)))

        self.resize_pop(new_pop_size)
        self.archive = self.archive[:new_arch_size]

        self.arch_size = new_arch_size
        self.curr_arch_size = min(self.curr_arch_size, self.arch_size)

        self.pop_size = new_pop_size

        self.update_memory_CrF()
        


