import os
import json
import ecole
import queue
import argparse
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        help='CUDA GPU id (-1 for CPU).',
        default=-1,
    )
    args = parser.parse_args()

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       ~~~~~~~~~~~~ CONFIG ~~~~~~~~~~~~~
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    # read default config file
    with open("config.default.json", 'r') as f:
        config = json.load(f)

    # gpu setup
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config['gpu']}"
        device = f"cuda:0"

    # torch setup
    import torch
    import torch.nn.functional as F
    from utilities import log
    from agent import Agent, AgentBrain

    if args.gpu > -1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Active CUDA Device: {torch.cuda.current_device()}")
    else:
        print("Running on CPU")

    # randomization setup
    rng = np.random.RandomState(config['seed'])
    torch.manual_seed(config['seed'])

    # working directory setup
    running_dir = f"../model/reinforced_agent/"
    os.makedirs(running_dir, exist_ok=True)

    # logging setup
    logfile = os.path.join(running_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)
    log(f'Learning rate: {config["lr"]}', logfile)
    log(f'Entropy bonus: {config["entropy_bonus"]}', logfile)
    log(f'Discount factor: {config["discount_factor"]}', logfile)
    log(f'Sample rate: {config["sample_rate"]}', logfile)
    log(f'Batch size: {config["batch_size"]}', logfile)

    # train on only one instance
    instance = 'instance.lp'

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       ~~~~~~~~~~~~~ TRAIN ~~~~~~~~~~~~~
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    jobs_queue = queue.Queue()
    brain = AgentBrain(config, device)
    agents = [Agent(j, config, brain, jobs_queue) for j in range(config['num_agents'])]
    for agent in agents:
        agent.start()

    best_valid_result = np.inf
    for epoch in range(config['num_epochs']):
        log(f'EPOCH {epoch}', logfile)

        # validation
        is_validation_epoch = (epoch % config['validate_every'] == 0)
        if is_validation_epoch:
            log('Started validation run', logfile)
            validation_results = []
            greedy = True
            for j in range(config['num_valid_seeds']):
                seed = rng.randint(low=0, high=2**16)
                task = (instance, seed, greedy, validation_results)
                jobs_queue.put(task)
            jobs_queue.join() # wait for valid jobs to be finished
            valid_result = np.mean(validation_results)
            log(f'Validation finished. Average tree size was: {valid_result}', logfile)
            if best_valid_result > valid_result:
                best_valid_result = valid_result
                log(f'Best result so far. Saving parameters.', logfile)
                brain.save(f'{running_dir}/best_params.pkl')


        # send train tasks
        samples = []
        greedy = False
        log('Started training', logfile)
        for j in range(config['num_episodes_per_epoch']):
            seed = rng.randint(low=0, high=2**16)
            task = (instance, seed, greedy, samples)
            jobs_queue.put(task)
        jobs_queue.join() # wait for train jobs to be finished
        log(f'Number of samples: {len(samples)}', logfile)

        # train on samples
        stats = brain.update(samples)
        log('Model updated: avg loss  %.2f .'%stats["loss"], logfile)


log('Training finished. Best result was %.2f'%best_valid_result, logfile)

# send stop signals to all agents
for j in range(config['num_agents']):
    task = (None, None, None, None)
    jobs_queue.put(task)
