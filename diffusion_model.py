""" A 1D diffusion model."""
import numpy as np
import matplotlib.pyplot as plt


# Start by setting two fixed model parameters, the diffusivity and the size of the model domain

# In[2]:


D=100
Lx=300


# Next, setup the model grid using a NUmpy array

# In[4]:


dx=0.5
x=np.arange(start=0,stop=Lx,step=dx)
nx=len(x)


# Set the initial conditions for the model.
# The cake 'C' is a step function with a high balue of the left, a low value on the tight, and a step at the center of the domain

# In[12]:


C=np.zeros_like(x)
C_left=500
C_right=0
C[x<=Lx/2]=C_left
C[x>Lx/2]=C_right


# Plot the initial profile

# In[13]:


plt.figure()
plt.plot(x,C,'r')
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial Profile")


# In[14]:


nt=5000
dt=0.5*dx**2/D


# Loop over the time steps of the model, solving the diffusion equation using the FTCS scheme shown above. Note the use of array operations on the variable `C`. The boundary conditions remain fixed om each time step

# In[15]:


for t in range(0, nt):
	C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])


# In[17]:


z=list(range(5))


# In[19]:


plt.figure()
plt.plot(x,C,'b')
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial Profile")


# In[ ]:




