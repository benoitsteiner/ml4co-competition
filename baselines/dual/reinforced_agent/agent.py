import os
import sys
import ecole
import threading
import numpy as np
from scipy.signal import lfilter

import torch
import torch.nn.functional as F
import torch_geometric
import utilities

sys.path.insert(0, os.path.abspath(f'../'))
from model.gcn_model import GNNPolicy


class Agent(threading.Thread):
    """
    Agent class.
    Receives solving tasks through the job queue and processes them using the
    current policy. If greedy is set to True, actions are chosen greedily and
    the agent returns the total tree size. Else, actions are taken stochastically
    and the agent returns samples from the solving process.

    Parameters
    ----------
    id_ : int
        Agent id.
    config : str
        Config file.
    brain : AgentBrain
        Reference to an AgentBrain holding the policy.
    jobs_queue : queue.Queue
        Queue from which to receive job orders.
    """
    def __init__(self, id_, config, brain, jobs_queue):
        super().__init__(name=id_)
        self.id = id_
        self.config = config
        self.brain = brain
        self.jobs_queue = jobs_queue

        self.rng = np.random.RandomState(id_)

        scip_params = {'separating/maxrounds': 0,
                       'presolving/maxrestarts': 0,
                       'limits/time': config['time_limit'],
                       'timing/clocktype': 2}

        self.environment = ecole.environment.Branching(
                            scip_params=scip_params,
                            observation_function=ecole.observation.NodeBipartite(),
                            reward_function=DualIntegral(),
                            information_function=ecole.reward.NNodes().cumsum()
        )

    def run(self):
        while True:
            instance, seed, greedy, results = self.jobs_queue.get()
            if instance is None: break

            self.environment.seed(seed)
            observation, action_set, reward, done, info = self.environment.reset(instance)

            counter = 0
            rewards = []
            samples = {}
            while not done:
                # step
                state = utilities.extract_state(observation, action_set)
                action = self.brain.sample_action(state, greedy)
                chosen_var = action_set[action]
                observation, action_set, reward, done, info = self.environment.step(chosen_var)

                # decide whether to keep the sample (we sub-sample trajectories)
                keep_sample = self.rng.rand() < self.config['sample_rate']
                if keep_sample and not greedy:
                    transition = utilities.Transition(state, action)
                    samples[counter] = transition

                # keep all rewards to calculate return
                rewards.append(reward)

                counter += 1

            # for training
            if not greedy:
                # calculate returns and attach them to their corresponding transition
                returns = self.get_discounted_return(rewards, self.config['discount_factor'])
                for idx in samples.keys():
                    samples[idx].returns = returns[idx]
                # send samples
                results.extend(samples.values())

            # for validation
            else:
                results.append(info)

            # singal that task is done
            self.jobs_queue.task_done()


    def get_discounted_return(self, rewards, discount):
        return lfilter([1], [1, -discount], rewards[::-1], axis=0)[::-1]




class AgentBrain:
    """
    Agent brain class.
    Holds a copy of the policy, which the agents query to pick an action. Also
    updates the policy using the samples provided by the agents.

    Parameters
    ----------
    config : str
        Config file.
    device : str
        Device on which tensors will be allocated.
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.policy = GNNPolicy().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config['lr'])
        self.random = np.random.RandomState(seed=self.config['seed'])

    def sample_action(self, state, greedy):
        """
        Sample an action given a state. If greedy, the action with highest
        probability is chosen. Otherwise, we sample from the distribution given
        by the policy

        Parameters
        ----------
        state : utilities.State
            Features describing the state.
        greedy : bool
            Whether action should be chosen greedily.

        Returns
        -------
        action : int
            Index of the chosen action in the action set list.
        """
        with torch.no_grad():
            state = state.to(self.device)
            logits = self.policy(state.constraint_features, state.edge_index, state.edge_attr, state.variable_features)
            if greedy:
                action = logits[state.action_set].argmax()
            else:
                policy = F.softmax(logits[state.action_set], dim=-1)
                action = self.random.choice(policy.shape[0], 1, p=policy.cpu().numpy())[0]
        return action.item()


    def update(self, transitions):
        """
        Update model parameters using sampled transitions.

        Parameters
        ----------
        transitions : list
            List of transitions coming from the agents.

        Returns
        -------
        stats : dict
            Some statistics about the update.
        """
        n_samples = len(transitions)
        transitions = torch_geometric.data.DataLoader(transitions, batch_size=32, shuffle=True)
        losses, reinforce_losses, entropies = [], [], []

        self.optimizer.zero_grad()
        for batch in transitions:
            batch = batch.to(self.device)
            loss = torch.tensor([0.0], device=self.device)
            logits = self.policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = utilities.pad_tensor(logits[batch.action_set], batch.action_set_size)
            dist = torch.distributions.categorical.Categorical(logits=logits)

            # reinforce loss
            returns = batch.returns.float()
            reinforce_loss = -(returns * dist.log_prob(batch.action)).sum()
            reinforce_loss /= n_samples
            loss += reinforce_loss

            # entropy regularizer
            entropy = dist.entropy().sum()
            entropy /= n_samples
            loss += - self.config['entropy_bonus']*entropy

            losses.append(loss.item()), reinforce_losses.append(reinforce_loss.item()), entropies.append(entropy.item())
            loss.backward()

        self.optimizer.step()

        stats = {'loss': np.sum(losses),
                 'entropy': np.sum(entropies),
                 'reinforce_loss': np.sum(reinforce_losses)}

        return stats

    def save(self, filename):
        """
        Saves policy parameters.

        Parameters
        ----------
        filename : str
            Name of the file where parameters are saved.
        """
        torch.save(self.policy.state_dict(), filename)



class DualIntegral:
    def __init__(self):
        self.previous_dual = None

    def before_reset(self, model):
        scip = model.as_pyscipopt()
        objective_sense = scip.getObjectiveSense()
        assert objective_sense == 'maximize'
        self.previous_dual = None

    def extract(self, model, done):
        scip = model.as_pyscipopt()
        current_bound = scip.getDualbound()

        if self.previous_dual is None:
            self.previous_dual = current_bound
            reward = None
        else:
            reward = - (current_bound-self.previous_dual)/2 - self.previous_dual
            self.previous_dual = current_bound

        #return reward
        return -1
