import numpy as np
#The following functions generate the interfaces to train and validate.
def generate_interfaces(Ninterfaces, length, zetas, amplitude = 1.0, shuffle = True):
    """
    Generates Ninterfaces of a certain length (integer) for each value of zeta (a float or list of floats) provided.
    Will return numpy arrays with type 'float32'.
    """
    if type(zetas)== float:
        zetas = [zetas]
    #
    q = 2*np.pi/length *( np.arange(Ninterfaces*length/2)%(length/2) + 1 )
    q = q.reshape(Ninterfaces, length//2)
    interfaces = []
    zeta_interfs = []
    for zeta in zetas:
        z_q = np.random.normal(scale=np.sqrt(amplitude*q**(-1-2*zeta))) * np.exp(1j * 2*np.pi*np.random.rand(length//2))
        u_z = np.fft.ifft(z_q, n=length).real
        u_z_quant = np.zeros_like(u_z)
        for interface in range(Ninterfaces):
            u_z_quant[interface] = np.round((u_z[interface] - u_z[interface].mean())*(length//2)) + length//2
            img = np.array([[1.]*length]*length) #1. is max brightness.
            for row in range(length):
                img[row, int(u_z_quant[interface,row]):] = 0.
            interfaces.append(img)
            zeta_interfs.append(zeta)
    if shuffle:
        indices = np.arange(len(interfaces))
        np.random.shuffle(indices)
        interfaces = np.array(interfaces)[indices]
        zeta_interfs = np.array(zeta_interfs)[indices]
    return tuple([interfaces.astype('float32'), zeta_interfs.astype('float32')])

def generate_train_validate_set(Ninterfaces, length, zetas, test_size):
    """
    Generate interfaces and split them into intrfs_train, intrfs_valid, zetas_train, zetas_valid.
    """
    all_train_interfaces, all_train_zetas = generate_interfaces(Ninterfaces, length, zetas)
    return train_test_split(all_train_interfaces, all_train_zetas, test_size=test_size)
