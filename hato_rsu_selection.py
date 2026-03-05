import math

def hato_rsu_selector(task_mips, rsu_dict):
    """
    Implements the RSU selection formulas from the 'Optimizing Energy-Efficient 
    Task Offloading in Edge Computing' research paper.
    """
    # (Alpha=Energy, Beta=Time, Gamma=Cost)
    ALPHA, BETA, GAMMA = 0.4, 0.4, 0.2
    
    # Constants for communication (Standard values used in the paper)
    P_TX = 0.1  # Transmission power (Watts)
    P_RX = 0.05 # Reception power (Watts)
    SIGMA = 1e-9 # Noise power
    
    best_rsu = None
    min_objective_value = float('inf')
    results = {}

    for rsu_id, rsu_data in rsu_dict.items():
        # 1. Calculate Transmission Rate (Shannon-Hartley Theorem)
        # Using bandwidth and a mock SINR (Signal Quality)
        bandwidth = rsu_data.get('bandwidth_mbps', 20) * 1e6 # Default 20MHz
        sinr = 25 # Signal-to-Interference-Noise Ratio in dB
        transmission_rate = bandwidth * math.log2(1 + sinr)

        # 2. Transmission Delay (T_trans)
        # S_i / R_i (Task size in bits / rate)
        # We approximate task bits as MIPS * 10^3 for this example
        task_bits = task_mips * 1000 
        t_trans = task_bits / transmission_rate

        # 3. Execution Delay at Edge (T_edge)
        # S_i / f_edge
        t_edge = task_mips / rsu_data['mips_available']

        # 4. Total Time (T_total)
        total_time = t_trans + t_edge

        # 5. Energy Consumption (E_total)
        # Formula from paper: P_tx * T_trans + P_edge * T_edge
        total_energy = (P_TX * t_trans) + (P_RX * t_edge)

        # 6. Objective Function (O_i)
        # Minimize (Alpha * E + Beta * T + Gamma * Cost)
        # We treat RAM availability as a proxy for 'Cost' (Higher RAM = Lower Cost/Risk)
        cost_proxy = 1 / rsu_data['ram_mb'] 
        objective_value = (ALPHA * total_energy) + (BETA * total_time) + (GAMMA * cost_proxy)

        results[rsu_id] = objective_value

        if objective_value < min_objective_value:
            min_objective_value = objective_value
            best_rsu = rsu_id

    return best_rsu, results