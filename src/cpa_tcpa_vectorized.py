# src/physics_engine.py
import numpy as np

def reconstruct_position(prev_pos, predicted_delta):
    """
    Module 2: Physics-Based Reconstruction.
    Applies the predicted displacement vector to the last known position.
    Formula: P(t+1) = P(t) + Delta_Predicted
    """
    return prev_pos + predicted_delta


def calculate_distance_error(pred_coords, true_coords):
    """
    Calculates the Haversine-approximate distance error in meters.
    Includes the cosine correction for Longitude based on Latitude.
    """
    # Latitude difference to meters (1 degree approx 111,000 meters)
    lat_diff = (pred_coords[:, 0] - true_coords[:, 0]) * 111000
    
    # Longitude difference with cosine correction at approx 38 degrees North (Piraeus)
    lon_diff = (pred_coords[:, 1] - true_coords[:, 1]) * 111000 * np.cos(np.deg2rad(38))
    
    # Euclidean distance of the error vectors
    error_meters = np.sqrt(lat_diff**2 + lon_diff**2)
    return error_meters


def calculate_cpa_tcpa(os_pos, os_vel, ts_pos, ts_vel):
    """
    Module 2: Deterministic CPA/TCPA Calculation (Chapter 4.4.1)
    
    Calculates the Closest Point of Approach (CPA) and Time to CPA (TCPA)
    using vectorized physics formulas.
    
    Args:
        os_pos: Own Ship position [lat, lon] in degrees
        os_vel: Own Ship velocity [vx, vy] in knots
        ts_pos: Target Ship position [lat, lon] in degrees
        ts_vel: Target Ship velocity [vx, vy] in knots
    
    Returns:
        cpa: Closest Point of Approach in nautical miles
        tcpa: Time to CPA in minutes (negative = diverging)
    """
    # Convert positions to nautical miles (approximate at Piraeus latitude)
    NM_PER_DEG_LAT = 60.0  # 1 degree latitude ≈ 60 nm
    NM_PER_DEG_LON = 60.0 * np.cos(np.deg2rad(38))  # Cosine correction
    
    # Relative position vector (Chapter 4.4.1)
    p_rel = np.array([
        (ts_pos[0] - os_pos[0]) * NM_PER_DEG_LAT,
        (ts_pos[1] - os_pos[1]) * NM_PER_DEG_LON
    ])
    
    # Relative velocity vector
    v_rel = np.array([
        ts_vel[0] - os_vel[0],
        ts_vel[1] - os_vel[1]
    ])
    
    # Calculate TCPA: -(P_rel · V_rel) / ||V_rel||²
    v_rel_squared = np.dot(v_rel, v_rel)
    
    if v_rel_squared < 1e-10:  # Vessels moving in parallel (edge case)
        tcpa = 0.0
        cpa = np.linalg.norm(p_rel)
    else:
        tcpa = -np.dot(p_rel, v_rel) / v_rel_squared
        
        # Calculate CPA: ||P_rel + TCPA · V_rel||
        closest_point = p_rel + tcpa * v_rel
        cpa = np.linalg.norm(closest_point)
    
    # Convert TCPA from hours to minutes (assuming velocity in knots = nm/hour)
    tcpa_minutes = tcpa * 60
    
    return cpa, tcpa_minutes


def calculate_cpa_tcpa_batch(os_positions, os_velocities, ts_positions, ts_velocities):
    """
    Vectorized batch calculation for multiple vessel pairs.
    
    Args:
        os_positions: Array of Own Ship positions (N, 2)
        os_velocities: Array of Own Ship velocities (N, 2)
        ts_positions: Array of Target Ship positions (N, 2)
        ts_velocities: Array of Target Ship velocities (N, 2)
    
    Returns:
        cpas: Array of CPA values in nautical miles
        tcpas: Array of TCPA values in minutes
    """
    n = len(os_positions)
    cpas = np.zeros(n)
    tcpas = np.zeros(n)
    
    for i in range(n):
        cpas[i], tcpas[i] = calculate_cpa_tcpa(
            os_positions[i], os_velocities[i],
            ts_positions[i], ts_velocities[i]
        )
    
    return cpas, tcpas