import numpy as np

class House():
    def __init__(self, K, C, Tout, Tin, Tset, occupancy, Qhvac, hvacON):
        self.K = K # thermal conductivity
        self.C = C # thermal capacity
        self.Tout = Tout # Outside Temperature
        self.Tin = Tin # Inside Temperature 
        self.Tset = Tset # Setpoint Temperature
        self.occupancy = occupancy # 0 (no one in the room) or 1 (somebody in the room)
        self.Qhvac = Qhvac # Cooling capacity
        self.Phvac = Qhvac # Electric power capacity 
        self.hvacON = hvacON # control action = 0 (OFF) or 1 (ON)
        self.timestep = 5/60.0 # 5 min simulation step converted to h
     
    def setup_schedule(
        self, days, hours, timestep, Tset_schedule, Tout_schedule,
        Tin_initial,
        ):

        self.number_of_days = days
        self.number_of_hours = hours
        self.timestep = timestep
        self.max_iterations = int((self.number_of_days*self.number_of_hours)*60 / self.timestep)

        self.Tset_schedule = Tset_schedule
        self.Tout_schedule = Tout_schedule
        self.Tin_initial = Tin_initial

    def update_Tout(self,Tout_new):
        self.Tout = Tout_new # Update to new outside temperature
        
    def update_Tset(self,Tset_new):
        self.Tset = Tset_new # Update to new setpoint temperature
    
    def update_hvacON(self,hvacONnew):
        self.hvacON = hvacONnew # update to new hvacON

    def update_occupancy(self,occupancy_new):
        self.occupancy = occupancy_new # update to new occupancy
      
    def update_Tin(self):
        """Update inside temperation.
        Describes the inside temperature evolution as a function of all other variables. 
        """

        self.Tin = self.Tin - self.timestep / self.C*(self.K*(self.Tin-self.Tout)+self.Qhvac*self.hvacON)

    def get_Power(self):
        COP = 3
        Power = self.Phvac*self.hvacON*COP 
        return Power
        
 # print the object nicely   
    def __str__(self):
        string_to_print = []
        for key in self.__dict__:
            string_to_print.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(string_to_print)
    
    def __repr__(self):
        return self.__str__() 

    def __iter__(self):
        self.iteration = 0
        return self
    
    def __next__(self):

        if self.iteration <= self.max_iterations - 4:
            Tset1 = self.Tset_schedule[self.iteration+1]
            Tset2 = self.Tset_schedule[self.iteration+2]
            Tset3 = self.Tset_schedule[self.iteration+3]
            Tout_next = self.Tout_schedule[self.iteration]
            self.iteration += 1
        else:
            StopIteration


def time_to_index(days, hours, timestep=5):
	hours_index = int(hours*60/timestep)
	days_index = int(days*24*60/timestep)
	return hours_index + days_index

