import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Polygon, Rectangle
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import io

def beam_fem_analysis(L, EI, ne, loads, supports, distLoads):
    """Main FEM analysis function that accepts parameters and returns results"""
    
    # ------------------------
    # FEM Computation
    nn = ne + 1
    le = L / ne
    x = np.linspace(0, L, nn)
    ndof = 2 * nn

    # Beam cross-section properties
    b = 0.3
    h = 0.6
    c = h/2
    I = b*h**3/12

    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # Assemble stiffness matrix
    for e in range(ne):
        idx = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
        k = (EI/le**3) * np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ])
        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += k[i, j]

    # Apply point loads
    if loads.size > 0:
        for i in range(loads.shape[0]):
            ni = np.argmin(np.abs(x - loads[i, 0]))
            F[2*ni] += loads[i, 1]  # The vertical DOF

    # Apply distributed loads if any
    if distLoads.size > 0:
        for i in range(distLoads.shape[0]):
            x1 = distLoads[i, 0]
            x2 = distLoads[i, 1]
            w = distLoads[i, 2]
            for e in range(ne):
                xe1 = x[e]
                xe2 = x[e+1]
                if xe1 >= x1 and xe2 <= x2:
                    idx = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
                    fe = w*le/2 * np.array([1, le/6, 1, -le/6])
                    F[idx] += fe

    # Apply boundary conditions
    fixed_dofs = []
    for i in range(len(supports)):
        ni = np.argmin(np.abs(x - supports[i]))
        fixed_dofs.append(2*ni)  # The vertical DOF

    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    U = np.zeros(ndof)
    
    # Solve the system
    U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])

    # Post-processing
    V = np.zeros(nn)
    M = np.zeros(nn)
    bending_stress = np.zeros(nn)
    shear_stress = np.zeros(nn)
    
    for e in range(ne):
        idx = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
        ue = U[idx]
        V[e] = (EI/le**3)*(12*ue[0]+6*le*ue[1]-12*ue[2]+6*le*ue[3])
        M[e] = (EI/le**2)*(6*ue[0]+4*le*ue[1]-6*ue[2]+2*le*ue[3])
        bending_stress[e] = M[e]*c/I
        shear_stress[e] = V[e]*c/I
    
    V[-1] = V[-2]
    M[-1] = M[-2]
    bending_stress[-1] = bending_stress[-2]
    shear_stress[-1] = shear_stress[-2]

    # Create analysis results figure
    fig = create_plot(x, M, V, bending_stress, shear_stress, L, loads, distLoads, supports, U)
    
    return fig, x, M, V, bending_stress, shear_stress, U

def create_plot(x, M, V, bending_stress, shear_stress, L, loads, distLoads, supports, U):
    """Create analysis result plots"""
    
    # Use a clean style
    mpl.style.use('default')
    
    # Create plot
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Plot 1: Bending Moment
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, M/1e3, 'b-o', linewidth=2, markersize=4)
    ax1.set_title('Bending Moment Diagram [kN·m]', fontweight='bold')
    ax1.set_ylabel('Moment [kN·m]')
    ax1.set_xlabel('Position along beam [m]')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.fill_between(x, M/1e3, 0, alpha=0.2, color='blue')
    
    # Plot 2: Shear Force
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, V/1e3, 'r-o', linewidth=2, markersize=4)
    ax2.set_title('Shear Force Diagram [kN]', fontweight='bold')
    ax2.set_ylabel('Shear Force [kN]')
    ax2.set_xlabel('Position along beam [m]')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.fill_between(x, V/1e3, 0, alpha=0.2, color='red')
    
    # Plot 3: Stress Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, bending_stress/1e6, 'm-', linewidth=2, label=r'$\sigma_b$')
    ax3.plot(x, shear_stress/1e6, 'g--', linewidth=2, label=r'$\tau$')
    ax3.legend(loc='best', frameon=True)
    ax3.set_title('Stress Distribution [MPa]', fontweight='bold')
    ax3.set_ylabel('Stress [MPa]')
    ax3.set_xlabel('Position along beam [m]')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Load Diagram - Histogram style
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Load Diagram [kN]', fontweight='bold')
    ax4.set_ylabel('Load [kN]')
    ax4.set_xlabel('x (m)')
    ax4.grid(True, linestyle='-', color='#E5E5E5', alpha=0.8)
    
    # Draw the beam line
    ax4.axhline(y=0, color='k', linewidth=2)
    
    # Plot distributed loads as histogram bars
    if distLoads.size > 0:
        for i in range(distLoads.shape[0]):
            x1 = distLoads[i, 0]
            x2 = distLoads[i, 1]
            w = distLoads[i, 2]/1e3  # Convert to kN
            
            # Add each segment as a rectangle (histogram bar)
            ax4.add_patch(Rectangle((x1, 0), 
                                   width=x2-x1, 
                                   height=w, 
                                   facecolor='#B0C4DE',  # Light blue color
                                   alpha=0.7, 
                                   edgecolor='#8A9CB3',  # Slightly darker edge
                                   linewidth=0.5))
    
    # Plot point loads with red arrows and text labels
    if loads.size > 0:
        for i in range(loads.shape[0]):
            xP = loads[i, 0]
            fP = loads[i, 1]/1e3  # Convert to kN
            
            # Draw arrow for point load
            arrow_length = 5  # Fixed arrow length
            ax4.arrow(xP, 0, 
                     0, -arrow_length if fP < 0 else arrow_length,
                     head_width=0.7, head_length=1.5, 
                     fc='red', ec='red', linewidth=1.5, zorder=10)
            
            # Add text label next to arrow
            offset_x = 0.5
            ax4.text(xP - 2, -5 if fP < 0 else 5, 
                    f"{fP:.1f} kN", 
                    color='red', ha='left', va='center', 
                    fontweight='bold', fontsize=9)
    
    # Draw support triangles
    for support in supports:
        triangle_height = 3
        triangle_width = 1.5
        triangle_x = [support-triangle_width/2, support, support+triangle_width/2]
        triangle_y = [0, -triangle_height, 0]
        ax4.fill(triangle_x, triangle_y, 'black', zorder=5)
    
    # Set y-axis limits
    max_load = 30
    if distLoads.size > 0:
        max_load = max(max_load, np.max(np.abs(distLoads[:, 2]/1e3)) * 1.2)
    
    y_max = max(120, max_load * 1.2)  # Set reasonable maximum
    ax4.set_ylim([-10, y_max])  # Leave space below zero for triangles
    ax4.set_xlim([0, L])
    
    # Set proper tick spacing
    ax4.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax4.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax4.tick_params(direction='in')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Add title to figure
    fig.suptitle('Beam FEM Analysis Results', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    return fig

def load_distributed_loads_from_excel(uploaded_file):
    """Process the uploaded Excel file for distributed loads in width/w format"""
    try:
        # Read the data
        df = pd.read_excel(uploaded_file)
        
        # Check if we have the expected columns
        expected_cols = ["width", "w"]
        if not all(col in df.columns for col in expected_cols):
            st.error(f'Excel must have columns: {", ".join(expected_cols)}')
            return np.array([])
        
        # Extract widths and loads
        widths = df["width"].values
        loads = df["w"].values
        
        # Calculate start and end positions by accumulating widths
        # Assuming widths are in mm, convert to m
        widths_m = widths / 1000.0  # mm to m
        
        # Calculate start positions
        start_positions = np.zeros(len(widths_m))
        for i in range(1, len(widths_m)):
            start_positions[i] = start_positions[i-1] + widths_m[i-1]
        
        # Calculate end positions
        end_positions = start_positions + widths_m
        
        # Create the distLoads array [start_pos, end_pos, load]
        distLoads = np.column_stack((start_positions, end_positions, loads))
        
        return distLoads
    except Exception as e:
        st.error(f"Error loading distributed loads: {e}")
        return np.array([])

def main():
    st.set_page_config(
        page_title="Beam FEM Analysis",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Beam FEM Analysis Tool")
    st.write("This application performs finite element analysis on a beam and visualizes the results.")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Beam Properties")
        
        # Calculate total beam length from data if available
        uploaded_file = st.file_uploader("Upload Excel file with distributed loads", type=["xlsx", "xls"])
        distLoads = np.array([])
        default_length = 72.0
        
        if uploaded_file is not None:
            distLoads = load_distributed_loads_from_excel(uploaded_file)
            if distLoads.size > 0:
                default_length = np.max(distLoads[:, 1])  # Use max end position as beam length
                st.success(f"Loaded {distLoads.shape[0]} distributed load segments")
                st.write("Preview:")
                preview_df = pd.DataFrame({
                    "Start [m]": distLoads[:, 0],
                    "End [m]": distLoads[:, 1],
                    "Load [N/m]": distLoads[:, 2]
                })
                st.dataframe(preview_df)
        
        L = st.slider("Beam Length [m]", 1.0, 100.0, float(default_length), 0.5)
        EI = st.number_input("Flexural Rigidity EI [N·m²]", min_value=1000.0, max_value=100000.0, value=20000.0, step=1000.0)
        ne = st.slider("Number of Elements", 5, 100, 36, 1)  # More elements for smoother results
        
        st.header("Support Locations")
        supports_text = st.text_input("Support Positions [m] (comma separated)", f"0, {L}")
        try:
            supports = np.array([float(x.strip()) for x in supports_text.split(',')])
        except:
            st.error("Invalid support format. Using default values.")
            supports = np.array([0, L])
        
        st.header("Point Loads")
        num_loads = st.number_input("Number of Point Loads", 0, 10, 2, 1)
        
        loads_data = []
        for i in range(num_loads):
            st.subheader(f"Point Load {i+1}")
            col1, col2 = st.columns(2)
            with col1:
                pos = st.number_input(f"Position [m]", 0.0, L, [3.0, 7.0][i] if i < 2 else L/2, 0.1, key=f"pos{i}")
            with col2:
                force = st.number_input(f"Force [N]", -100000.0, 100000.0, [-10000.0, -5000.0][i] if i < 2 else -1000.0, 100.0, key=f"force{i}")
            loads_data.append([pos, force])
        
        loads = np.array(loads_data) if loads_data else np.array([])
    
    # Main area
    if st.button("Run Analysis"):
        with st.spinner("Calculating..."):
            fig, x, M, V, bending_stress, shear_stress, U = beam_fem_analysis(L, EI, ne, loads, supports, distLoads)
            
            # Display plot
            st.pyplot(fig)
            
            # Create tabs for numerical results
            tab1, tab2, tab3, tab4 = st.tabs(["Displacement", "Moment & Shear", "Stress", "Export Data"])
            
            with tab1:
                displacement_df = pd.DataFrame({
                    'Position [m]': x,
                    'Displacement [m]': U[::2]
                })
                st.dataframe(displacement_df)
                
                # Plot displacement
                fig_disp, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x, U[::2], 'b-o')
                ax.set_xlabel('Position [m]')
                ax.set_ylabel('Displacement [m]')
                ax.set_title('Beam Displacement')
                ax.grid(True)
                st.pyplot(fig_disp)
            
            with tab2:
                moment_shear_df = pd.DataFrame({
                    'Position [m]': x,
                    'Bending Moment [kN·m]': M/1e3,
                    'Shear Force [kN]': V/1e3
                })
                st.dataframe(moment_shear_df)
            
            with tab3:
                stress_df = pd.DataFrame({
                    'Position [m]': x,
                    'Bending Stress [MPa]': bending_stress/1e6,
                    'Shear Stress [MPa]': shear_stress/1e6
                })
                st.dataframe(stress_df)
            
            with tab4:
                # Export to CSV
                csv = io.StringIO()
                
                export_df = pd.DataFrame({
                    'Position [m]': x,
                    'Displacement [m]': U[::2],
                    'Bending Moment [kN·m]': M/1e3,
                    'Shear Force [kN]': V/1e3,
                    'Bending Stress [MPa]': bending_stress/1e6,
                    'Shear Stress [MPa]': shear_stress/1e6
                })
                
                export_df.to_csv(csv, index=False)
                csv_data = csv.getvalue()
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name="beam_analysis_results.csv",
                    mime="text/csv",
                )
                
                # Export figure as PDF
                buf = io.BytesIO()
                fig.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="Download Plot as PDF",
                    data=buf,
                    file_name="beam_analysis_plot.pdf",
                    mime="application/pdf",
                )
                
    else:
        st.info("Upload data and click 'Run Analysis' to start the calculation.")
    
    # Footer
    st.markdown("---")
    st.caption("Beam FEM Analysis Tool | Developed using Streamlit and Python")

if __name__ == "__main__":
    main()