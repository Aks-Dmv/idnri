import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

class ChargedParticlesSim(object):
    def __init__(self, n_balls=3, box_size=2., loc_std=0.5, vel_norm=0.5,
                 interaction_strength=0.4, noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return np.abs(dist)

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=5100, sample_freq=100,
                          charge_prob=[1. / 2, 0., 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = np.zeros((T_save, self.n_balls, self.n_balls))
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog

            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            edges[counter] = (2.*(l2_dist_power3 >1.0)-1.)
            push_pull = -1.*(6.*(edges[counter] >0.)-1.)
            forces_size = self.interaction_strength * push_pull / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1
                    if counter == T_save:
                        break
                    edges[counter] = (2.*(l2_dist_power3 >1.0)-1.)

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                
                push_pull = -1.*(6.*(edges[counter] >0.)-1.)
                forces_size = self.interaction_strength * push_pull / l2_dist_power3
                # print(forces_size, edges[counter], l2_dist_power3)
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges


if __name__ == '__main__':
    sim = ChargedParticlesSim()

    for i in range(5):
            
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=5100, sample_freq=100)
        # print(edges)

        print("Simulation time: {}".format(time.time() - t))
        vel_norm = np.sqrt((vel ** 2).sum(axis=1))

        # for i in range(loc.shape[-1]):
        #     plt.plot(loc[:, 0, i], loc[:, 1, i])
        #     plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        fig, ax = plt.subplots()
        def update(frame):
            ax.clear()
            ax.plot(loc[frame, 0, 0], loc[frame, 1, 0], 'bo')
            ax.plot(loc[frame, 0, 1], loc[frame, 1, 1], 'ro')
            ax.plot(loc[frame, 0, 2], loc[frame, 1, 2], 'go')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        ani = animation.FuncAnimation(fig, update, interval=100, frames=50)
        ani.save("movie"+str(i)+".mp4", writer=writer)#, codec='mpeg4')
        # # plt.figure()
        # # energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges[i]) for i in
        # #             range(loc.shape[0])]
        # # # plt.plot(energies)
        # plt.show()
