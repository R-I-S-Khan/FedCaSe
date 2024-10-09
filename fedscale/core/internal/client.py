import os
import csv

class Client(object):

    def __init__(self, hostId, clientId, speed, traces=None):
        self.hostId = hostId
        self.clientId = clientId
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.sample_size = speed['sample_size']
        self.score = 0
        self.traces = traces
        self.behavior_index = 0
        self.samples_in_cache = 0
        self.clientpath = os.path.join('/home/cc', 'client')

    def update_samples_in_cache(self,num_samples):
        self.samples_in_cache = num_samples

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def isActive(self, cur_time):
        if self.traces is None:
            return True

        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True

        return False

    def read_samples_map(self,file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header 
            for row in reader:
                total = float(row[0])
                cache = float(row[1])
                ssd = float(row[2])
                #samptoimp_map[samp] = imp
        return total, cache, ssd

    def getCompletionTime(self, batch_size, upload_step, upload_size, download_size, augmentation_factor=3.0):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers, 
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        # return (3.0 * batch_size * num_steps/float(self.compute_speed) + model_size/float(self.bandwidth))
        #avg_img_size = 0.6 #avg size of image of femnist dataset in kilobyte (Kb)
        avg_img_size = self.sample_size * 1024 #avg size of image of femnist dataset in kilobyte (Kb)
        #microsd_speed = 0.0001 #avg time in seconds required to read 1 Kb from microsd card. #0.0001 #0.15
        #ram_speed = 0.00000005 #avg time in seconds required to read 1 Kb from RAM #0.00000005 #0.00001

        microsd_speed = 0.0001 #avg time in seconds required to read 1 Kb from microsd card. #0.0001 #0.15
        ram_speed = 0.00000005 #avg time in seconds required to read 1 Kb from RAM #0.00000005 #0.00001
        tot_samps = batch_size * upload_step
        from_ssd = tot_samps - self.samples_in_cache

        self.samples_path = os.path.join(self.clientpath, str(self.clientId) + '_samples.csv')

        if os.path.exists(self.samples_path) and os.path.isfile(self.samples_path):
            total, cache, ssd = self.read_samples_map(self.samples_path)
        else:
            print(f"clientId: {self.clientId} first time.")
            total = batch_size * upload_step
            cache = 0.0
            ssd = total

        # checking theoretical upper limit improvement.
        # calc_total = batch_size * upload_step
        # print(f"clientId: {self.clientId}, total: {total}, calc_total: {calc_total}")

        #temporary addition for warmup
        # total = batch_size * upload_step
        # cache = total
        # ssd = 0

        calc_total = batch_size * upload_step
        #print(f"clientId: {self.clientId}, total: {total}, calc_total: {calc_total}")

        #print(f"earlier: clientId: {self.clientId}, total samples: {tot_samps}, from_cache: {self.samples_in_cache}, from_ssd: {from_ssd}")
        #print(f"current: clientId: {self.clientId}, total samples: {total}, from_cache: {cache}, from_ssd: {ssd}")

        # return {'computation': augmentation_factor * batch_size * upload_step*float(self.compute_speed)/1000.,
        #         'communication': (upload_size+download_size)/float(self.bandwidth),
        #         'io': ((batch_size *upload_step) - self.samples_in_cache)*avg_img_size*microsd_speed  + self.samples_in_cache * avg_img_size * ram_speed }


        comp = augmentation_factor * total *float(self.compute_speed)/1000
        comm = (upload_size+download_size)/float(self.bandwidth)


        io = (ssd*avg_img_size*microsd_speed)  + (cache * avg_img_size * ram_speed)

        total_time = comp + comm + io
        
        frac_comp = (comp / total_time)*100
        frac_comm = (comm / total_time)*100
        frac_io = (io/total_time)*100

        print(f"clientId: {self.clientId}, frac_comp: {frac_comp}, frac_comm: {frac_comm}, frac_io: {frac_io}, avg_img_size: {self.sample_size}")

        return {'computation': augmentation_factor * total *float(self.compute_speed)/1000.,
                'communication': (upload_size+download_size)/float(self.bandwidth),
                'io': (ssd*avg_img_size*microsd_speed)  + (cache * avg_img_size * ram_speed) }
        # return (augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000. + \
        #         (upload_size+download_size)/float(self.bandwidth))
