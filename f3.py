import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import hashlib
import yaml
from yaml.loader import SafeLoader
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import base64
from io import BytesIO
import subprocess
import webbrowser
from enum import Enum
from typing import Dict
import uuid

# --------------------------
# Constants and Configuration
class TaskStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

# --------------------------
# UI Styling
# --------------------------
def set_page_styles():
    st.markdown("""
    <style>
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --accent: #4776E6;
        --dark: #1a1a2e;
        --light: #f8f9fa;
        --success: #4BB543;
        --warning: #FFA500;
        --danger: #FF5252;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background: url('https://images.unsplash.com/photo-1635070041078-e363dbe005cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
    }
    
    .login-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        width: 100%;
        max-width: 450px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .login-title {
        color: var(--primary);
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(to right, #6e48aa, #4776E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .colored-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .gradient-card {
        position: relative;
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    
    .algorithm-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .superposition-tag {
        background-color: #e0f2fe;
        color: #0369a1;
    }
    
    .genetic-tag {
        background-color: #f0fdf4;
        color: #15803d;
    }
    
    .annealing-tag {
        background-color: #fef2f2;
        color: #b91c1c;
    }
    
    .hybrid-tag {
        background-color: #f5f3ff;
        color: #7c3aed;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-online {
        background-color: var(--success);
    }
    
    /* New styles for enhanced UI */
    .feature-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .webpage-card {
        border-left: 4px solid var(--primary);
        padding: 1rem;
        margin-bottom: 0.5rem;
        background: white;
        border-radius: 0.5rem;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .user-profile {
        text-align: center;
        padding: 1.5rem;
    }
    
    /* Dark mode toggle */
    .dark-mode .main {
        background: #121212;
        color: #ffffff;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .login-box {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# Authentication
# --------------------------
def setup_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    if not os.path.exists('config.yaml'):
        default_config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@quantumalloc.com',
                        'name': 'Admin',
                        'password': hashlib.sha256('admin123'.encode()).hexdigest(),
                        'role': 'admin',
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    'analyst': {
                        'email': 'analyst@quantumalloc.com',
                        'name': 'Analyst',
                        'password': hashlib.sha256('analyst123'.encode()).hexdigest(),
                        'role': 'analyst',
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    'imran': {
                        'email': 'thoubicimran@gmail.com',
                        'name': 'Imran',
                        'password': hashlib.sha256('imran005'.encode()).hexdigest(),
                        'role': 'superuser',
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'quantum_alloc_auth',
                'name': 'quantum_alloc_auth'
            },
            'preauthorized': {
                'emails': ['admin@quantumalloc.com', 'analyst@quantumalloc.com', 'thoubicimran@gmail.com']
            }
        }
        with open('config.yaml', 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    def login_form():
        st.markdown("""
        <div class="login-container">
            <div class="login-box">
                <h1 class="login-title">Quantum Allocation</h1>
                <p class="login-subtitle">Optimized Quantum Task Distribution System</p>
        """, unsafe_allow_html=True)
        
        # Tab system for login/signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                input_username = st.text_input("Username", key="input_username")
                input_password = st.text_input("Password", type="password", key="input_password")
                submitted = st.form_submit_button("Login", type="primary")
                
                if submitted:
                    if input_username in config['credentials']['usernames']:
                        stored_pw = config['credentials']['usernames'][input_username]['password']
                        if stored_pw == hashlib.sha256(input_password.encode()).hexdigest():
                            st.session_state.authenticated = True
                            st.session_state.current_user = input_username
                            st.session_state.user_email = config['credentials']['usernames'][input_username]['email']
                            st.session_state.name = config['credentials']['usernames'][input_username]['name']
                            st.session_state.user_role = config['credentials']['usernames'][input_username].get('role', 'user')
                            st.rerun()
                        else:
                            st.error("Invalid password")
                    else:
                        st.error("Username not found")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a username")
                new_email = st.text_input("Email address")
                new_name = st.text_input("Full name")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_submitted = st.form_submit_button("Create Account")
                
                if signup_submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif new_username in config['credentials']['usernames']:
                        st.error("Username already exists")
                    else:
                        # Add new user to config
                        config['credentials']['usernames'][new_username] = {
                            'email': new_email,
                            'name': new_name,
                            'password': hashlib.sha256(new_password.encode()).hexdigest(),
                            'role': 'user',
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        # Save updated config
                        with open('config.yaml', 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                        st.success("Account created successfully! Please login.")
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.authenticated:
        login_form()
        st.stop()
    
    with st.sidebar:
        # User profile with dark mode toggle
        st.markdown(f"""
        <div class="user-profile">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, var(--primary), var(--secondary)); 
                border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                {st.session_state.name[0].upper()}
            </div>
            <h3 style="margin-bottom: 0.5rem;">{st.session_state.name}</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">{st.session_state.user_email}</p>
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
                <span class="status-indicator status-online"></span>
                <span style="color: var(--success); font-size: 0.8rem;">Online</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dark mode toggle
        st.session_state.dark_mode = st.toggle("Dark Mode", st.session_state.dark_mode)
        
        if st.button("🚪 Logout", type="primary", key="logout_button", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.name = None
            st.session_state.user_email = None
            st.session_state.user_role = None
            st.rerun()
    
    return st.session_state.current_user

# --------------------------
# System Configuration
# --------------------------
def get_system_config():
    return {
        'devices': ['Quantum Processor 1', 'Quantum Processor 2', 'Quantum Processor 3'],
        'capacities': [50, 50, 50],
        'costs': [1.2, 2.5, 3.8],
        'latencies': [0.1, 0.2, 0.3],
        'reliability': [0.99, 0.95, 0.90],
        'status': ['Online', 'Online', 'Maintenance']
    }

# --------------------------
# Task Configuration
# --------------------------
def get_task_config():
    """Generate and return task configuration"""
    if 'task_config' not in st.session_state:
        num_tasks = st.session_state.get('num_tasks_slider', 8)
        st.session_state.task_config = {
            'num_tasks': num_tasks,
            'tasks': [generate_task(i) for i in range(num_tasks)]
        }
    return st.session_state.task_config

def update_task_config():
    """Update task configuration based on current settings"""
    num_tasks = st.session_state.get('num_tasks_slider', 8)
    st.session_state.task_config = {
        'num_tasks': num_tasks,
        'tasks': [generate_task(i) for i in range(num_tasks)]
    }

def generate_task(task_index: int) -> Dict:
    """Generate a single task with proper ID"""
    task_type = random.choice(['computation', 'simulation', 'optimization'])
    
    task = {
        "task_id": f"Task_{task_index + 1}",  # 1-based ID
        "priority": random.choice([1, 2, 3]),
        "status": TaskStatus.PENDING.value,
        "type": task_type,
        "size": random.randint(10, 40),
        "deadline": round(random.uniform(1.0, 5.0), 2)
    }
    
    # Add web launch details for some tasks
    if random.random() < 0.3:  # 30% chance to be a web launch task
        web_category = random.choice(list(WEB_DATABASE.keys()))
        web_name = random.choice(list(WEB_DATABASE[web_category].keys()))
        task.update({
            "type": "web_launch",
            "web_name": web_name,
            "web_url": WEB_DATABASE[web_category][web_name]["url"]
        })
    
    return task

def validate_allocation(allocation, expected_task_count):
    """Validate that allocation contains all expected tasks"""
    missing_tasks = [
        f"Task_{i+1}" 
        for i in range(expected_task_count) 
        if f"Task_{i+1}" not in allocation
    ]
    if missing_tasks:
        raise ValueError(f"Missing tasks in allocation: {missing_tasks}")
    return True

# --------------------------
# Algorithm Implementations
# --------------------------
def measure_time(func, *args):
    """Measure execution time of a function"""
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, round(end - start, 4)

def execute_web_launch(task: Dict) -> bool:
    """Execute the web page launch task"""
    try:
        webbrowser.open_new_tab(task["web_url"])
        return True
    except Exception as e:
        st.error(f"Failed to open {task['web_name']}: {str(e)}")
        return False

def quantum_superposition(system_config, task_config):
    devices = system_config['devices']
    device_capacities = system_config['capacities']
    device_costs = system_config['costs']
    device_latencies = system_config['latencies']
    device_reliability = system_config['reliability']
    
    allocation = {}
    priority_weights = np.array([task['priority'] for task in task_config['tasks']]) / sum([task['priority'] for task in task_config['tasks']])
    capacity_factors = np.array([1/(c+0.1) for c in device_capacities])
    capacity_factors = capacity_factors / capacity_factors.sum()
    
    probabilities = np.outer(priority_weights, capacity_factors)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    for i, task in enumerate(task_config['tasks']):
        device_index = np.random.choice(range(len(devices)), p=probabilities[i])
        allocation[task['task_id']] = {
            'size': task['size'],
            'device': devices[device_index],
            'priority': task['priority'],
            'deadline': task['deadline'],
            'assigned_cost': round(task['size'] * device_costs[device_index], 2),
            'expected_latency': round(device_latencies[device_index], 3),
            'reliability': device_reliability[device_index],
            'type': task['type'],
            **({'web_name': task['web_name'], 'web_url': task['web_url']} 
               if task.get('type') == 'web_launch' else {})
        }
    return allocation

def quantum_genetic_algorithm(system_config, task_config):
    devices = system_config['devices']
    device_capacities = system_config['capacities']
    device_costs = system_config['costs']
    device_latencies = system_config['latencies']
    device_reliability = system_config['reliability']
    
    population_size = 20
    generations = 100
    
    def fitness(solution):
        load = {device: 0 for device in devices}
        for i, device in enumerate(solution):
            load[device] += task_config['tasks'][i]['size']
        
        overload = sum([max(0, load[device] - device_capacities[devices.index(device)]) * 10 for device in devices])
        total_cost = sum([device_costs[devices.index(device)] * task_config['tasks'][i]['size'] for i, device in enumerate(solution)])
        total_latency = sum([device_latencies[devices.index(device)] * task_config['tasks'][i]['priority'] for i, device in enumerate(solution)])
        
        return -(overload + total_cost + total_latency)
    
    population = [[random.choice(devices) for _ in range(len(task_config['tasks']))] for _ in range(population_size)]
    
    for _ in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        selected = population[:population_size // 2]
        next_generation = selected.copy()
        
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            crossover_point = random.randint(1, len(task_config['tasks'])-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            
            if random.random() < 0.1:
                child[random.randint(0, len(task_config['tasks'])-1)] = random.choice(devices)
            next_generation.append(child)
        
        population = next_generation

    best_solution = max(population, key=fitness)
    
    allocation = {}
    for i, task in enumerate(task_config['tasks']):
        device = best_solution[i]
        device_index = devices.index(device)
        allocation[task['task_id']] = {
            'size': task['size'],
            'device': device,
            'priority': task['priority'],
            'deadline': task['deadline'],
            'assigned_cost': round(task['size'] * device_costs[device_index], 2),
            'expected_latency': round(device_latencies[device_index], 3),
            'reliability': device_reliability[device_index],
            'type': task['type'],
            **({'web_name': task['web_name'], 'web_url': task['web_url']} 
               if task.get('type') == 'web_launch' else {})
        }
    
    return allocation

def quantum_annealing(system_config, task_config):
    devices = system_config['devices']
    device_capacities = system_config['capacities']
    device_costs = system_config['costs']
    device_latencies = system_config['latencies']
    device_reliability = system_config['reliability']
    
    def energy(solution):
        load = {device: 0 for device in devices}
        for i, device in enumerate(solution):
            load[device] += task_config['tasks'][i]['size']
        
        overload = sum([max(0, load[device] - device_capacities[devices.index(device)]) * 10 for device in devices])
        total_cost = sum([device_costs[devices.index(device)] * task_config['tasks'][i]['size'] for i, device in enumerate(solution)])
        total_latency = sum([device_latencies[devices.index(device)] * task_config['tasks'][i]['priority'] for i, device in enumerate(solution)])
        
        return overload + total_cost + total_latency
    
    current_solution = [random.choice(devices) for _ in range(len(task_config['tasks']))]
    current_energy = energy(current_solution)
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    temperature = 1000
    cooling_rate = 0.95
    
    for _ in range(500):
        neighbor = current_solution.copy()
        idx = random.randint(0, len(task_config['tasks'])-1)
        neighbor[idx] = random.choice(devices)
        
        neighbor_energy = energy(neighbor)
        
        if neighbor_energy < current_energy or random.random() < np.exp((current_energy - neighbor_energy) / temperature):
            current_solution = neighbor
            current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy
    
        temperature *= cooling_rate

    allocation = {}
    for i, task in enumerate(task_config['tasks']):
        device = best_solution[i]
        device_index = devices.index(device)
        allocation[task['task_id']] = {
            'size': task['size'],
            'device': device,
            'priority': task['priority'],
            'deadline': task['deadline'],
            'assigned_cost': round(task['size'] * device_costs[device_index], 2),
            'expected_latency': round(device_latencies[device_index], 3),
            'reliability': device_reliability[device_index],
            'type': task['type'],
            **({'web_name': task['web_name'], 'web_url': task['web_url']} 
               if task.get('type') == 'web_launch' else {})
        }
    
    return allocation

def hybrid_quantum_optimization(system_config, task_config):
    # Start with genetic algorithm
    ga_result = quantum_genetic_algorithm(system_config, task_config)
    
    # Use the result as initial state for annealing
    initial_solution = [ga_result[f"Task_{i+1}"]['device'] for i in range(len(task_config['tasks']))]
    
    # Run annealing with better initial state
    devices = system_config['devices']
    device_capacities = system_config['capacities']
    device_costs = system_config['costs']
    device_latencies = system_config['latencies']
    device_reliability = system_config['reliability']
    
    def energy(solution):
        load = {device: 0 for device in devices}
        for i, device in enumerate(solution):
            load[device] += task_config['tasks'][i]['size']
        
        overload = sum([max(0, load[device] - device_capacities[devices.index(device)]) * 10 for device in devices])
        total_cost = sum([device_costs[devices.index(device)] * task_config['tasks'][i]['size'] for i, device in enumerate(solution)])
        total_latency = sum([device_latencies[devices.index(device)] * task_config['tasks'][i]['priority'] for i, device in enumerate(solution)])
        
        return overload + total_cost + total_latency
    
    current_solution = initial_solution
    current_energy = energy(current_solution)
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    temperature = 500
    cooling_rate = 0.95
    
    for _ in range(300):
        neighbor = current_solution.copy()
        idx = random.randint(0, len(task_config['tasks'])-1)
        neighbor[idx] = random.choice(devices)
        
        neighbor_energy = energy(neighbor)
        
        if neighbor_energy < current_energy or random.random() < np.exp((current_energy - neighbor_energy) / temperature):
            current_solution = neighbor
            current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy
    
        temperature *= cooling_rate

    allocation = {}
    for i, task in enumerate(task_config['tasks']):
        device = best_solution[i]
        device_index = devices.index(device)
        allocation[task['task_id']] = {
            'size': task['size'],
            'device': device,
            'priority': task['priority'],
            'deadline': task['deadline'],
            'assigned_cost': round(task['size'] * device_costs[device_index], 2),
            'expected_latency': round(device_latencies[device_index], 3),
            'reliability': device_reliability[device_index],
            'type': task['type'],
            **({'web_name': task['web_name'], 'web_url': task['web_url']} 
               if task.get('type') == 'web_launch' else {})
        }
    
    return allocation

# --------------------------
# Web Page Database
# --------------------------
WEB_DATABASE = {
    "Social Media": {
        "Facebook": {"url": "https://www.facebook.com", "icon": "📘"},
        "Twitter": {"url": "https://twitter.com", "icon": "🐦"},
        "Instagram": {"url": "https://www.instagram.com", "icon": "📷"},
        "LinkedIn": {"url": "https://www.linkedin.com", "icon": "💼"},
        "Reddit": {"url": "https://www.reddit.com", "icon": "🔴"}
    },
    "Productivity": {
        "Gmail": {"url": "https://mail.google.com", "icon": "✉️"},
        "Google Drive": {"url": "https://drive.google.com", "icon": "🗄️"},
        "Google Docs": {"url": "https://docs.google.com", "icon": "📝"},
        "Google Sheets": {"url": "https://sheets.google.com", "icon": "📊"},
        "Google Slides": {"url": "https://slides.google.com", "icon": "📑"}
    },
    "Entertainment": {
        "YouTube": {"url": "https://www.youtube.com", "icon": "▶️"},
        "Netflix": {"url": "https://www.netflix.com", "icon": "🎬"},
        "Spotify": {"url": "https://www.spotify.com", "icon": "🎵"},
        "Twitch": {"url": "https://www.twitch.tv", "icon": "🎮"},
        "Disney+": {"url": "https://www.disneyplus.com", "icon": "🏰"}
    },
    "News": {
        "BBC": {"url": "https://www.bbc.com", "icon": "🇬🇧"},
        "CNN": {"url": "https://www.cnn.com", "icon": "🇺🇸"},
        "Al Jazeera": {"url": "https://www.aljazeera.com", "icon": "🌍"},
        "Reuters": {"url": "https://www.reuters.com", "icon": "📰"},
        "The Guardian": {"url": "https://www.theguardian.com", "icon": "🛡️"}
    }
}

CUSTOM_PAGES = {
    "My Portfolio": {"url": "https://example.com/portfolio", "icon": "🌟"},
    "Project Dashboard": {"url": "https://example.com/dashboard", "icon": "📈"},
    "Company Site": {"url": "https://example.com/company", "icon": "🏢"},
    "Internal Wiki": {"url": "https://example.com/wiki", "icon": "📚"},
    "Team Chat": {"url": "https://example.com/chat", "icon": "💬"}
}

# --------------------------
# Web Launcher Functions
# --------------------------
def web_launcher_page():
    MAX_CONCURRENT_TABS = 10
    
    # Initialize session state
    if 'opened_pages' not in st.session_state:
        st.session_state.opened_pages = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'custom_pages' not in st.session_state:
        st.session_state.custom_pages = CUSTOM_PAGES.copy()
    
    def launch_web_page(page_name: str, page_data: dict, from_favorites=False) -> bool:
        """Launch a web page with tab limit enforcement"""
        if len(st.session_state.opened_pages) >= MAX_CONCURRENT_TABS:
            st.warning(f"Cannot open {page_name}. Maximum {MAX_CONCURRENT_TABS} tabs allowed.")
            return False
        
        try:
            webbrowser.open_new_tab(page_data["url"])
            st.session_state.opened_pages.append({
                "name": page_name,
                "url": page_data["url"],
                "icon": page_data["icon"],
                "open_time": time.time()
            })
            
            # Add to history
            st.session_state.history.append({
                "name": page_name,
                "url": page_data["url"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            if not from_favorites:
                st.success(f"Opened {page_name} successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to open {page_name}: {str(e)}")
            return False
    
    def launch_all_pages():
        """Launch all web pages respecting tab limits"""
        successful_launches = 0
        total_pages = sum(len(pages) for pages in WEB_DATABASE.values()) + len(st.session_state.custom_pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Launch standard pages
        for category, pages in WEB_DATABASE.items():
            for page_name, page_data in pages.items():
                if len(st.session_state.opened_pages) >= MAX_CONCURRENT_TABS:
                    status_text.warning(f"Stopped: Reached maximum of {MAX_CONCURRENT_TABS} tabs")
                    break
                
                status_text.info(f"Opening {page_name}...")
                if launch_web_page(page_name, page_data):
                    successful_launches += 1
                
                progress_bar.progress(successful_launches / total_pages)
                time.sleep(0.3)
        
        # Launch custom pages
        for page_name, page_data in st.session_state.custom_pages.items():
            if len(st.session_state.opened_pages) >= MAX_CONCURRENT_TABS:
                status_text.warning(f"Stopped: Reached maximum of {MAX_CONCURRENT_TABS} tabs")
                break
            
            status_text.info(f"Opening {page_name}...")
            if launch_web_page(page_name, page_data):
                successful_launches += 1
            
            progress_bar.progress(successful_launches / total_pages)
            time.sleep(0.3)
        
        status_text.success(f"Opened {successful_launches} of {total_pages} pages")
        progress_bar.empty()
    
    def close_all_pages():
        """Close all opened web pages (simulated)"""
        if not st.session_state.opened_pages:
            st.warning("No web pages currently opened")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_pages = len(st.session_state.opened_pages)
        
        for i in range(total_pages, 0, -1):
            page = st.session_state.opened_pages.pop()
            status_text.info(f"Closing {page['name']}...")
            progress_bar.progress((total_pages - i + 1) / total_pages)
            time.sleep(0.2)
        
        status_text.success("All web pages closed successfully!")
        progress_bar.empty()
        time.sleep(2)
        status_text.empty()
    
    def close_page(index: int):
        """Simulate closing a web page"""
        try:
            page = st.session_state.opened_pages.pop(index)
            st.success(f"Closed {page['name']}")
        except Exception as e:
            st.error(f"Error closing page: {str(e)}")
    
    def toggle_favorite(page_name, page_url):
        """Add or remove page from favorites"""
        if any(fav['name'] == page_name for fav in st.session_state.favorites):
            st.session_state.favorites = [fav for fav in st.session_state.favorites if fav['name'] != page_name]
            st.success(f"Removed {page_name} from favorites")
        else:
            st.session_state.favorites.append({
                "name": page_name,
                "url": page_url,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success(f"Added {page_name} to favorites")
    
    # Web launcher page UI
    st.title("🌐 Web Page Launcher")
    st.write(f"Maximum concurrent tabs allowed: {MAX_CONCURRENT_TABS}")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Open All Pages", help="Open all web pages up to the tab limit"):
            launch_all_pages()
            st.rerun()
    
    with col2:
        if st.button("🛑 Close All Pages", help="Close all currently opened pages"):
            close_all_pages()
            st.rerun()
    
    # Display opened pages
    with st.expander("🌐 Currently Opened Pages", expanded=True):
        if not st.session_state.opened_pages:
            st.info("No web pages currently opened")
        else:
            for i, page in enumerate(st.session_state.opened_pages):
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.write(f"{page['icon']} **{page['name']}** (open for {int(time.time() - page['open_time'])}s)")
                with col2:
                    if st.button("Close", key=f"close_{i}"):
                        close_page(i)
                        st.rerun()
                with col3:
                    if st.button("⭐", key=f"fav_{i}"):
                        toggle_favorite(page['name'], page['url'])
                        st.rerun()
    
    # Tab system for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Standard Pages", "Custom Pages", "Favorites", "History"])

    with tab1:
        st.header("Standard Web Pages")
        for category, pages in WEB_DATABASE.items():
            with st.expander(f"📁 {category}"):
                cols = st.columns(3)
                for i, (page_name, page_data) in enumerate(pages.items()):
                    with cols[i % 3]:
                        if st.button(f"{page_data['icon']} {page_name}", key=f"launch_{page_name}"):
                            launch_web_page(page_name, page_data)
                            st.rerun()
    
    with tab2:
        st.header("🔧 Custom Web Pages")
        cols = st.columns(3)
        for i, (page_name, page_data) in enumerate(st.session_state.custom_pages.items()):
            with cols[i % 3]:
                if st.button(f"{page_data['icon']} {page_name}", key=f"custom_{page_name}"):
                    launch_web_page(page_name, page_data)
                    st.rerun()
        
        # Add custom page form
        with st.expander("➕ Add New Custom Page"):
            with st.form("add_custom_page"):
                col1, col2 = st.columns(2)
                with col1:
                    page_name = st.text_input("Page Name")
                    page_url = st.text_input("URL (include https://)")
                with col2:
                    page_icon = st.text_input("Icon (emoji)", value="🔗")
                    st.markdown("Find emojis: [Emoji Cheat Sheet](https://www.webfx.com/tools/emoji-cheat-sheet/)")
                
                if st.form_submit_button("Add Page"):
                    if page_name and page_url and page_icon:
                        if page_name not in st.session_state.custom_pages:
                            st.session_state.custom_pages[page_name] = {"url": page_url, "icon": page_icon}
                            st.success(f"Added {page_name} to custom pages!")
                        else:
                            st.error("A page with this name already exists")
                    else:
                        st.error("Please fill all fields")
    
    with tab3:
        st.header("⭐ Favorite Pages")
        if not st.session_state.favorites:
            st.info("No favorite pages yet")
        else:
            cols = st.columns(3)
            for i, fav in enumerate(st.session_state.favorites):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div class="webpage-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4>{fav['name']}</h4>
                                <button onclick="window.open('{fav['url']}', '_blank')" style="background: none; border: none; cursor: pointer;">➡️</button>
                            </div>
                            <p style="color: #666; font-size: 0.8rem;">{fav['url']}</p>
                            <div style="display: flex; justify-content: space-between;">
                                <button onclick="toggleFavorite('{fav['name']}')" style="background: none; border: none; cursor: pointer;">⭐</button>
                                <small style="color: #999;">Added: {fav['timestamp']}</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab4:
        st.header("🕒 Browsing History")
        if not st.session_state.history:
            st.info("No browsing history yet")
        else:
            # Group history by date
            history_by_date = {}
            for item in reversed(st.session_state.history):
                date = item['timestamp'].split()[0]
                if date not in history_by_date:
                    history_by_date[date] = []
                history_by_date[date].append(item)
            
            for date, items in history_by_date.items():
                with st.expander(f"📅 {date}"):
                    for item in items:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{item['name']}**")
                            st.caption(item['url'])
                        with col2:
                            if st.button("Open", key=f"hist_{item['name']}_{item['timestamp']}"):
                                launch_web_page(item['name'], {"url": item['url'], "icon": "⏳"}, from_favorites=True)
                                st.rerun()

# --------------------------
# Algorithm Comparison
# --------------------------
def algorithm_comparison_page(system_config, task_config):
    st.markdown("""
    <div class="colored-header">
        <h1>Algorithm Performance Comparison</h1>
        <p>Compare all optimization approaches side-by-side</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔁 Run All Algorithms", type="primary", use_container_width=True):
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = run_all_algorithms(system_config, task_config)
        show_comparison_results(st.session_state.comparison_results)
    elif 'comparison_results' in st.session_state:
        show_comparison_results(st.session_state.comparison_results)
    else:
        st.info("Click the button above to run all algorithms and compare their performance")

def run_all_algorithms(system_config, task_config):
    results = {}
    
    with st.spinner("Running Quantum Superposition..."):
        superposition_result, exec_time = measure_time(quantum_superposition, system_config, task_config)
        results['superposition'] = {
            'allocation': superposition_result,
            'exec_time': exec_time,
            'total_cost': sum(task['assigned_cost'] for task in superposition_result.values()),
            'avg_latency': np.mean([task['expected_latency'] for task in superposition_result.values()]),
            'avg_reliability': np.mean([task['reliability'] for task in superposition_result.values()]),
            'max_load': calculate_max_load(system_config, superposition_result),
            'deadline_met': calculate_deadline_met(system_config, superposition_result),
            'efficiency': calculate_efficiency(system_config, superposition_result)
        }
    
    with st.spinner("Running Quantum Genetic Algorithm..."):
        genetic_result, exec_time = measure_time(quantum_genetic_algorithm, system_config, task_config)
        results['genetic'] = {
            'allocation': genetic_result,
            'exec_time': exec_time,
            'total_cost': sum(task['assigned_cost'] for task in genetic_result.values()),
            'avg_latency': np.mean([task['expected_latency'] for task in genetic_result.values()]),
            'avg_reliability': np.mean([task['reliability'] for task in genetic_result.values()]),
            'max_load': calculate_max_load(system_config, genetic_result),
            'deadline_met': calculate_deadline_met(system_config, genetic_result),
            'efficiency': calculate_efficiency(system_config, genetic_result)
        }
    
    with st.spinner("Running Quantum Annealing..."):
        annealing_result, exec_time = measure_time(quantum_annealing, system_config, task_config)
        results['annealing'] = {
            'allocation': annealing_result,
            'exec_time': exec_time,
            'total_cost': sum(task['assigned_cost'] for task in annealing_result.values()),
            'avg_latency': np.mean([task['expected_latency'] for task in annealing_result.values()]),
            'avg_reliability': np.mean([task['reliability'] for task in annealing_result.values()]),
            'max_load': calculate_max_load(system_config, annealing_result),
            'deadline_met': calculate_deadline_met(system_config, annealing_result),
            'efficiency': calculate_efficiency(system_config, annealing_result)
        }
    
    with st.spinner("Running Hybrid Quantum Optimization..."):
        hybrid_result, exec_time = measure_time(hybrid_quantum_optimization, system_config, task_config)
        results['hybrid'] = {
            'allocation': hybrid_result,
            'exec_time': exec_time,
            'total_cost': sum(task['assigned_cost'] for task in hybrid_result.values()),
            'avg_latency': np.mean([task['expected_latency'] for task in hybrid_result.values()]),
            'avg_reliability': np.mean([task['reliability'] for task in hybrid_result.values()]),
            'max_load': calculate_max_load(system_config, hybrid_result),
            'deadline_met': calculate_deadline_met(system_config, hybrid_result),
            'efficiency': calculate_efficiency(system_config, hybrid_result)
        }
    
    return results

def calculate_max_load(system_config, allocation):
    """Calculate the maximum load percentage across devices"""
    device_loads = {device: 0 for device in system_config['devices']}
    for task in allocation.values():
        device_loads[task['device']] += task['size']
    
    max_load = 0
    for i, device in enumerate(system_config['devices']):
        capacity = system_config['capacities'][i]
        load_pct = (device_loads[device] / capacity) * 100
        if load_pct > max_load:
            max_load = load_pct
    return round(max_load, 2)

def calculate_deadline_met(system_config, allocation):
    """Calculate percentage of tasks that meet their deadlines"""
    total_tasks = len(allocation)
    if total_tasks == 0:
        return 0
    
    met_count = 0
    for task in allocation.values():
        device_index = system_config['devices'].index(task['device'])
        latency = system_config['latencies'][device_index]
        if latency <= task['deadline']:
            met_count += 1
    return round((met_count / total_tasks) * 100, 2)

def calculate_efficiency(system_config, allocation):
    """Calculate overall efficiency score (0-100)"""
    if not allocation:
        return 0
    
    # Cost efficiency (lower is better)
    max_cost = max(system_config['costs']) * sum(task['size'] for task in allocation.values())
    actual_cost = sum(task['assigned_cost'] for task in allocation.values())
    cost_score = (1 - (actual_cost / max_cost)) * 100 if max_cost > 0 else 100
    
    # Load balance score (lower max load is better)
    load_score = (100 - calculate_max_load(system_config, allocation))
    
    # Deadline score
    deadline_score = calculate_deadline_met(system_config, allocation)
    
    # Reliability score
    reliability_score = np.mean([task['reliability'] for task in allocation.values()]) * 100
    
    # Weighted average
    return round((cost_score * 0.3 + load_score * 0.2 + deadline_score * 0.3 + reliability_score * 0.2), 2)

def show_comparison_results(results):
    st.markdown("""
    <style>
    .gradient-card {
        color: #333333 !important;
        font-weight: 600 !important;
    }
    .gradient-card .metric-label {
        font-size: 0.75rem;
        color: #666666;
        font-weight: 600;
    }
    .gradient-card .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #000000;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="colored-header">
        <h1>Algorithm Performance Comparison</h1>
        <p>Detailed metrics across all optimization approaches</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Performance Metrics")
    cols = st.columns(4)
    
    for i, (algo, data) in enumerate(results.items()):
        with cols[i]:
            algo_class = {
                'superposition': 'superposition-tag',
                'genetic': 'genetic-tag',
                'annealing': 'annealing-tag',
                'hybrid': 'hybrid-tag'
            }[algo]
            
            st.markdown(f"""
            <div class="gradient-card">
                <span class="algorithm-tag {algo_class}" style="font-weight: 700;">{algo.capitalize()}</span>
                <div style="margin-top: 1rem;">
                    <div class="metric-label">Execution Time</div>
                    <div class="metric-value">{data['exec_time']}s</div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="metric-label">Total Cost</div>
                    <div class="metric-value">${data['total_cost']:.2f}</div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="metric-label">Avg Latency</div>
                    <div class="metric-value">{data['avg_latency']:.3f}ms</div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="metric-label">Avg Reliability</div>
                    <div class="metric-value">{data['avg_reliability']:.1%}</div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="metric-label">Max Load</div>
                    <div class="metric-value">{data['max_load']}%</div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="metric-label">Deadline Met</div>
                    <div class="metric-value">{data['deadline_met']}%</div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="metric-label">Efficiency</div>
                    <div class="metric-value">{data['efficiency']}/100</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("📈 Comparative Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Execution Time", "Cost Efficiency", "Reliability", "Load Balance", "Efficiency"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(results.keys()),
            y=[data['exec_time'] for data in results.values()],
            marker_color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
        ))
        fig.update_layout(title='Execution Time Comparison (seconds)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(results.keys()),
            y=[data['total_cost'] for data in results.values()],
            marker_color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
        ))
        fig.update_layout(title='Total Cost Comparison ($)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(results.keys()),
            y=[data['avg_reliability'] for data in results.values()],
            marker_color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
        ))
        fig.update_layout(title='Average Reliability Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(results.keys()),
            y=[data['max_load'] for data in results.values()],
            marker_color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
        ))
        fig.update_layout(title='Maximum Device Load (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(results.keys()),
            y=[data['efficiency'] for data in results.values()],
            marker_color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
        ))
        fig.update_layout(title='Overall Efficiency Score (0-100)')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detailed Metrics Comparison")
    
    # Prepare data for the table
    metrics_data = []
    for algo, data in results.items():
        metrics_data.append({
            "Algorithm": algo.capitalize(),
            "Execution Time (s)": data['exec_time'],
            "Total Cost ($)": f"{data['total_cost']:.2f}",
            "Avg Latency (ms)": f"{data['avg_latency']:.3f}",
            "Avg Reliability": f"{data['avg_reliability']:.1%}",
            "Max Load (%)": f"{data['max_load']}",
            "Deadline Met (%)": f"{data['deadline_met']}",
            "Efficiency Score": f"{data['efficiency']}/100"
        })

    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Display as HTML table with custom styling
    st.markdown(df.style
        .set_properties(**{
            'background-color': '#ffffff',
            'color': '#333333',
            'border-color': '#dddddd',
            'font-weight': '600'
        })
        .set_table_styles([{
            'selector': 'th',
            'props': [('background-color', '#6e48aa'), 
                     ('color', 'white'),
                     ('font-weight', 'bold')]
        }])
        .hide(axis="index")
        .to_html(), unsafe_allow_html=True)

    # Add download button for the comparison data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Comparison Data",
        data=csv,
        file_name="algorithm_comparison.csv",
        mime="text/csv"
    )

# --------------------------
# Task Allocation Page
# --------------------------
def task_allocation_page(system_config, task_config):
    st.markdown("""
    <div class="colored-header">
        <h1>Quantum-Inspired Task Allocation System</h1>
        <p>Optimize task distribution across quantum processors</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Task Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_num_tasks = st.slider(
                "Number of tasks", 
                min_value=3, 
                max_value=20, 
                value=task_config['num_tasks'], 
                key="num_tasks_slider"
            )
            
            if st.button("🔄 Generate New Tasks"):
                update_task_config()
                st.rerun()
        
        with col2:
            st.markdown("**Task Priority Distribution**")
            priority_counts = {
                "High (3)": sum(1 for task in task_config['tasks'] if task['priority'] == 3),
                "Medium (2)": sum(1 for task in task_config['tasks'] if task['priority'] == 2),
                "Low (1)": sum(1 for task in task_config['tasks'] if task['priority'] == 1)
            }
            st.bar_chart(priority_counts)

    st.subheader("📋 Task Information")
    cols = st.columns(4)
    cols[0].metric("Total Tasks", task_config['num_tasks'])
    cols[1].metric("Total Size", f"{sum(task['size'] for task in task_config['tasks'])} units")
    cols[2].metric("Avg Priority", f"{np.mean([task['priority'] for task in task_config['tasks']]):.1f}")
    cols[3].metric("Avg Deadline", f"{np.mean([task['deadline'] for task in task_config['tasks']]):.1f}s")

    with st.expander("🔍 View Detailed Task Information"):
        task_details = []
        for task in task_config['tasks']:
            task_details.append({
                "Task ID": task['task_id'],
                "Size": task['size'],
                "Priority": task['priority'],
                "Type": f"{task['type']} ({task['web_name']})" if task.get('web_name') else task['type'],
                "Deadline (s)": f"{task['deadline']:.2f}",
                "Criticality": "High" if task['priority'] == 3 else "Medium" if task['priority'] == 2 else "Low"
            })
        st.dataframe(task_details, use_container_width=True, hide_index=True)

    st.subheader("⚡ Optimization Configuration")
    algorithm = st.selectbox(
        "Select Optimization Algorithm",
        ("Quantum Superposition", "Quantum Genetic Algorithm", 
         "Quantum Annealing", "Hybrid Quantum Optimization"),
        index=0
    )

    if st.button("🚀 Run Optimization", use_container_width=True, type="primary"):
        with st.spinner(f"Running {algorithm} optimization..."):
            if algorithm == "Quantum Superposition":
                allocation, exec_time = measure_time(quantum_superposition, system_config, task_config)
            elif algorithm == "Quantum Genetic Algorithm":
                allocation, exec_time = measure_time(quantum_genetic_algorithm, system_config, task_config)
            elif algorithm == "Quantum Annealing":
                allocation, exec_time = measure_time(quantum_annealing, system_config, task_config)
            else:
                allocation, exec_time = measure_time(hybrid_quantum_optimization, system_config, task_config)

        st.success(f"Optimization completed in {exec_time} seconds")
        
        st.subheader("📊 Allocation Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cost", f"${sum(task['assigned_cost'] for task in allocation.values()):.2f}")
        m2.metric("Avg Latency", f"{np.mean([task['expected_latency'] for task in allocation.values()]):.3f} ms")
        m3.metric("Avg Reliability", f"{np.mean([task['reliability'] for task in allocation.values()]):.1%}")
        m4.metric("Tasks Allocated", task_config['num_tasks'])

        st.subheader("📈 Device Utilization")
        device_stats = {device: {'load': 0, 'tasks': 0, 'cost': 0} for device in system_config['devices']}
        for task in allocation.values():
            device = task['device']
            device_stats[device]['load'] += task['size']
            device_stats[device]['tasks'] += 1
            device_stats[device]['cost'] += task['assigned_cost']
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]])
        fig.add_trace(
            go.Bar(
                x=system_config['devices'],
                y=[device_stats[d]['load'] for d in system_config['devices']],
                name='Load',
                marker_color=['#636EFA', '#EF553B', '#00CC96']
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Pie(
                labels=system_config['devices'],
                values=[device_stats[d]['tasks'] for d in system_config['devices']],
                name='Task Distribution',
                marker_colors=['#636EFA', '#EF553B', '#00CC96']
            ),
            row=1, col=2
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detailed Allocation")
        allocation_data = []
        for task_id, task_info in allocation.values():
            allocation_data.append({
                "Task ID": task_id,
                "Size": task_info['size'],
                "Priority": task_info['priority'],
                "Device": task_info['device'],
                "Cost": f"${task_info['assigned_cost']:.2f}",
                "Latency": f"{task_info['expected_latency']} ms",
                "Reliability": f"{task_info['reliability']:.1%}",
                "Deadline": f"{task_info['deadline']:.2f}s",
                "Type": task_info['type']
            })
        
        st.dataframe(allocation_data, use_container_width=True, hide_index=True)

        csv = pd.DataFrame(allocation_data).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Allocation Report",
            data=csv,
            file_name="quantum_allocation_report.csv",
            mime="text/csv"
        )

# --------------------------
# Main Application
# --------------------------
def main():
    set_page_styles()
    username = setup_auth()
    
    system_config = get_system_config()
    task_config = get_task_config()
    
    # Display system info in sidebar
    st.sidebar.markdown("### System Configuration")
    sys_info = pd.DataFrame({
        "Device": system_config['devices'],
        "Capacity": system_config['capacities'],
        "Cost/Unit": system_config['costs'],
        "Latency (ms)": system_config['latencies'],
        "Reliability": [f"{r:.0%}" for r in system_config['reliability']],
        "Status": system_config['status']
    })
    st.sidebar.dataframe(sys_info, use_container_width=True, hide_index=True)
    
    if st.session_state.user_role == 'admin':
        with st.sidebar.expander("⚙️ System Config Editor"):
            st.write("System configuration editor coming soon")
    
    # Unified navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Task Allocation", "Algorithm Comparison", "Web Page Launcher"],
        label_visibility="collapsed"
    )
    
    if page == "Task Allocation":
        task_allocation_page(system_config, task_config)
    elif page == "Algorithm Comparison":
        algorithm_comparison_page(system_config, task_config)
    elif page == "Web Page Launcher":
        web_launcher_page()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.875rem;">
        <p>Quantum-Inspired Task Allocation System | Developed with Streamlit</p>
        <p>© 2023 Quantum Allocation Technologies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
