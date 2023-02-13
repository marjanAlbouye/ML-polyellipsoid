import torch.nn as nn


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act_fn="ReLU"):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn

        self.force_net = nn.Sequential(*self._get_net())
        self.torque_net = nn.Sequential(*self._get_net())

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Dropout(p=0.5))
            layers.append(self._get_act_fn())
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return layers

    def forward(self, x):
        return self.force_net(x), self.torque_net(x)


class EllipsCustomForce(hoomd.md.force.Custom):
    def __init__(self, rigid_ids, model_path, in_dim, hidden_dim, out_dim, n_layers, act_fn):
        super().__init__(aniso=True)
        # load ML model
        self.rigid_ids = rigid_ids
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NN(in_dim=in_dim, hidden_dim=hidden_dim,
                        out_dim=out_dim, n_layers=n_layers, act_fn=act_fn)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def set_forces(self, timestep):
        # get positions and orientations
        with self._state.cpu_local_snapshot as snap:
            rigid_rtags = snap.particles.rtag[self.rigid_ids]
            positions = np.array(snap.particles.position[rigid_rtags], copy=True)
            orientations = np.array(snap.particles.orientation[rigid_rtags], copy=True)

        positions_tensor = torch.from_numpy(positions).type(torch.FloatTensor)
        orientations_tensor = torch.from_numpy(orientations).type(torch.FloatTensor)
        model_input = torch.cat((positions_tensor, orientations_tensor), 1).to(self.device)
        force_prediction, torque_prediction = self.model(model_input)
        predicted_force = force_prediction.cpu().detach().numpy()
        predicted_torque = torque_prediction.cpu().detach().numpy()
        with self.cpu_local_force_arrays as arrays:
            # print('****************************************')
            # print('timestep: ', timestep)
            # print('predicted force: ', predicted_force)
            # print('predicted torque: ', predicted_torque)
            # arrays.force[self.rigid_ids] = predicted_force
            # arrays.torque[self.rigid_ids] = predicted_torque

            arrays.force[rigid_rtags] = predicted_force
            arrays.torque[rigid_rtags] = predicted_torque