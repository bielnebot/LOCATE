# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:27:27 2021

@author: Jose M. Alsina (UPC)
"""

from parcels import rng as random
import math


def AdvectionRK4(particle, fieldset, time):
    
    if particle.beached == 0:
        
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        #particle.beached = 2
        
def AdvectionRK4_un(particle, fieldset, time):
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
     

def BrownianMotion2D(particle, fieldset, time):
    if particle.beached == 0:
    #if particle.beached == 0:
        kh_meridional = fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon]
        kh_zonal = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon]
        dx = fieldset.meshSize[time, particle.depth, particle.lat, particle.lon]
        dx0 = 1000

        particle.lat += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(particle.dt) * kh_meridional * math.pow(dx/dx0, 1.33))
        particle.lon += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(particle.dt) * kh_zonal      * math.pow(dx/dx0, 1.33))
        #particle.beached = 3

def DiffusionUniformKh_custom(particle,fieldset,time):
    if particle.beached == 0:
    #if particle.beached==0:

        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy

def StokesDrag(particle, fieldset, time):
    #if particle.beached == 0:
    if particle.beached == 0:
        (u_uss, v_uss) = fieldset.UVuss[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_uss * particle.dt
        particle.lat += v_uss * particle.dt

        
            
def Unbeaching(particle, fieldset, time):
    if particle.beached == 1:
        # if the random resuspension time set is < age (ie if the age is larger than this resuspension variable)
        
        if (particle.resus_time*86400) <= particle.age: 
            resus_prob=math.exp(-particle.dt/(particle.resus_time*86400.))
            if particle.resus_rand > resus_prob:
                (ub, vb) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
                particle.lon += ub * particle.dt
                particle.lat += vb * particle.dt
                particle.beached = 2
                #particle.unbeachCount += 1 
                
            else:
                particle.beached = 1


# particle bounces back to the position directly before being beached (dependent on dt (5 mins))    
def Bounce(particle, fieldset, time):
    if particle.beached == 1:
        (ubo, vbo) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        # unsure whether this if statement is required, the particle is already beached
        if ubo == 0 and vbo == 0:  # with u,v at the last position
            particle.lon = particle.prev_lon
            particle.lat = particle.prev_lat 

          
# https://github.com/OceanParcels/parcels/discussions/1071
def distance_trajectory(particle, fieldset, time):
    # Calculate the distance in latitudinal direction (using 1.11e2 kilometer per degree latitude)
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    # Calculate the distance in longitudinal direction, using cosine(latitude) - spherical earth
    lon_dist = (particle.lon - particle.prev_lon) * 1.11e2 * math.cos(particle.lat * math.pi / 180)
    # Calculate the total Euclidean distance travelled by the particle
    particle.distance_traj += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))
    # Set the stored values for next iteration, this is for the previous dt
    particle.prev_lon = particle.lon  
    particle.prev_lat = particle.lat
   
            
def Ageing(particle, fieldset, time):
    particle.age += particle.dt


def DeleteParticle(particle, fieldset, time):
    print("Particle [%d] lost !! (%g %g %g %g)" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()
    

def distance_shore(particle, fieldset, time):
    if particle.beached == 0:
        # gets the velocity fields of UV.
        # v is always 0, therefore not included
        U_dist = fieldset.U_dist[time, particle.depth, particle.lat, particle.lon]
        # conversion of m/s of u and m/s of v
        particle.distance_shore = U_dist
        #particle.u_field = U_dist        

def beaching_distance(particle, fieldset, time): 
    # distance to shore
    if particle.beached == 0:       
        if particle.distance_shore <= 0:
            particle.beached = 1
            particle.delete()
        else:
            particle.beached = 0

# legacy beaching dependent on current velocity
def beaching_velocity(particle, fieldset, time): 
    #velocity
    if particle.beached == 0:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-16 and fabs(v) < 1e-16:
            particle.beached = 1
            particle.delete()
        else:
            particle.beached = 0 
    
# distance based on mean velocity of IBI files used for the period 1/2/17 to 19/10/17
# if particle is > 6 hours at a distance < 1.694 km
def beaching_proximity(particle, fieldset, time): 
    if particle.beached == 0:
        if particle.distance_shore <= 1.694:
            particle.proximity += particle.dt
        else:
            particle.proximity = 0
        
        if particle.proximity > 21600:
            particle.beached = 1
            particle.delete()
        else:
            particle.beached = 0