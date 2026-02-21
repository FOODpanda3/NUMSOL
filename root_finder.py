import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class RootFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods: Root Finder Lab")
        self.root.geometry("1250x700")

        # --- STEP 2: FRONTEND LAYOUT (The Interface) ---
        # Initialize Style for Table Zooming
        self.style = ttk.Style()
        self.current_font_size = 10
        self.update_table_style() 

        # Bind Enter Key
        self.root.bind('<Return>', lambda event: self.on_calculate())

        # UI Layout Construction
        self.setup_input_panel()    # Top Frame
        self.setup_results_area()   # Bottom Frames (Table & Graph)

        # --- NEW: Variables for Dynamic Graphing ---
        self.current_f = None       # To store the active lambda function
        self.main_line = None       # To store the blue plot line object

    # ---------------------------------------------------------
    # STEP 1: BACKEND LOGIC (The Algorithms)
    # ---------------------------------------------------------
    def evaluate_function(self, expr, x_val):
        """Safely parses user string input and calculates f(x)."""
        safe_dict = {
            "x": x_val, "np": np, 
            "sin": np.sin, "cos": np.cos, "tan": np.tan, 
            "exp": np.exp, "log": np.log, "log10": np.log10, 
            "sqrt": np.sqrt, "square": np.square, 
            "abs": np.abs, "pi": np.pi, "e": np.e
        }
        try:
            return eval(expr, {"__builtins__": None}, safe_dict)
        except:
            return None

    def bisection(self, f, xl, xu, tol, max_iter):
        """Returns: Final Root, History List"""
        history = []
        if f(xl) * f(xu) >= 0: return None, []
        xr = xl
        for i in range(1, max_iter + 1):
            xr_old = xr
            xr = (xl + xu) / 2
            error = abs((xr - xr_old) / xr) * 100 if xr != 0 else 0
            # History structure: Iter, p1, p2, root, root^2, error
            history.append((i, f"{xl:.5f}", f"{xu:.5f}", f"{xr:.5f}", f"{xr**2:.5f}", f"{error:.5f}%"))
            if f(xl) * f(xr) < 0: xu = xr
            else: xl = xr
            if error < tol: break
        return xr, history

    def false_position(self, f, xl, xu, tol, max_iter):
        """Returns: Final Root, History List"""
        history = []
        if f(xl) * f(xu) >= 0: return None, []
        xr = xl
        for i in range(1, max_iter + 1):
            xr_old = xr
            fxl = f(xl)
            fxu = f(xu)
            if (fxl - fxu) == 0: break
            xr = xu - (fxu * (xl - xu)) / (fxl - fxu)
            error = abs((xr - xr_old) / xr) * 100 if xr != 0 else 0
            history.append((i, f"{xl:.5f}", f"{xu:.5f}", f"{xr:.5f}", f"{xr**2:.5f}", f"{error:.5f}%"))
            if f(xl) * f(xr) < 0: xu = xr
            else: xl = xr
            if error < tol: break
        return xr, history

    def newton_raphson(self, f, x0, tol, max_iter):
        """Returns: Final Root, History List"""
        history = []
        xi = x0
        h = 1e-5
        for i in range(1, max_iter + 1):
            f_xi = f(xi)
            f_prime = (f(xi + h) - f(xi - h)) / (2 * h)
            if abs(f_prime) < 1e-10: return None, []
            xi_next = xi - (f_xi / f_prime)
            error = abs((xi_next - xi) / xi_next) * 100 if xi_next != 0 else 0
            history.append((i, "-", "-", f"{xi_next:.5f}", f"{xi_next**2:.5f}", f"{error:.5f}%"))
            xi = xi_next
            if error < tol: break
        return xi, history

    def secant(self, f, x0, x1, tol, max_iter):
        """Returns: Final Root, History List"""
        history = []
        for i in range(1, max_iter + 1):
            f_x0 = f(x0)
            f_x1 = f(x1)
            if abs(f_x1 - f_x0) < 1e-10: return None, [] 
            x2 = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
            error = abs((x2 - x1) / x2) * 100 if x2 != 0 else 0
            history.append((i, f"{x0:.5f}", f"{x1:.5f}", f"{x2:.5f}", f"{x2**2:.5f}", f"{error:.5f}%"))
            x0 = x1
            x1 = x2
            if error < tol: break
        return x1, history

    # ---------------------------------------------------------
    # STEP 2: FRONTEND LAYOUT HELPERS
    # ---------------------------------------------------------
    def update_table_style(self):
        self.style.configure("Custom.Treeview", 
                             font=('Helvetica', self.current_font_size), 
                             rowheight=int(self.current_font_size * 2.5))
        self.style.configure("Custom.Treeview.Heading", 
                             font=('Helvetica', self.current_font_size, 'bold'))

    def setup_input_panel(self):
        # Top Frame: Labels, Entries, Buttons
        input_frame = ttk.LabelFrame(self.root, text="Configuration")
        input_frame.pack(side="top", fill="x", padx=10, pady=5)

        ttk.Label(input_frame, text="f(x):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_func = ttk.Entry(input_frame, width=25)
        self.entry_func.insert(0, "x**2 - 24")
        self.entry_func.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Method:").grid(row=0, column=2, padx=5, sticky="e")
        self.method_var = tk.StringVar()
        self.method_menu = ttk.Combobox(input_frame, textvariable=self.method_var, state="readonly", width=15)
        self.method_menu['values'] = ("Bisection", "False Position", "Newton-Raphson", "Secant")
        self.method_menu.current(0)
        self.method_menu.grid(row=0, column=3, padx=5)
        self.method_menu.bind("<<ComboboxSelected>>", self.toggle_inputs)

        self.lbl_p1 = ttk.Label(input_frame, text="Lower (xl) / x0:")
        self.lbl_p1.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.entry_p1 = ttk.Entry(input_frame, width=10)
        self.entry_p1.insert(0, "2")
        self.entry_p1.grid(row=1, column=1, sticky="w", padx=5)

        self.lbl_p2 = ttk.Label(input_frame, text="Upper (xu) / x1:")
        self.lbl_p2.grid(row=1, column=2, padx=5, sticky="e")
        self.entry_p2 = ttk.Entry(input_frame, width=10)
        self.entry_p2.insert(0, "6")
        self.entry_p2.grid(row=1, column=3, sticky="w", padx=5)

        ttk.Label(input_frame, text="Tolerance:").grid(row=1, column=4, padx=5, sticky="e")
        self.entry_tol = ttk.Entry(input_frame, width=10)
        self.entry_tol.insert(0, "0.0001")
        self.entry_tol.grid(row=1, column=5, sticky="w", padx=5)

        ttk.Label(input_frame, text="Max Iter:").grid(row=1, column=6, padx=5, sticky="e")
        self.entry_max_iter = ttk.Entry(input_frame, width=8)
        self.entry_max_iter.insert(0, "100")
        self.entry_max_iter.grid(row=1, column=7, sticky="w", padx=5)

        self.btn_calc = ttk.Button(input_frame, text="CALCULATE\n(Enter)", command=self.on_calculate)
        self.btn_calc.grid(row=0, column=8, rowspan=2, padx=15, sticky="ns")

    def setup_results_area(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=5)

        # Bottom-Left Frame: Table
        table_frame = ttk.LabelFrame(bottom_frame, text="Iteration Table (Ctrl+Scroll to Zoom)")
        table_frame.pack(side="left", fill="both", expand=True, padx=5)

        columns = ("iter", "p1", "p2", "root", "sq_root", "error")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Custom.Treeview")
        
        headers = ["Iter", "Lower / x0", "Upper / x1", "Root (x)", "Root² (Check)", "Error (%)"]
        for col, h in zip(columns, headers):
            self.tree.heading(col, text=h)
            self.tree.column(col, width=90, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind Table Zoom
        self.tree.bind("<Control-MouseWheel>", self.on_table_zoom)
        self.tree.bind("<Control-Button-4>", lambda e: self.on_table_zoom(e, direction=1))
        self.tree.bind("<Control-Button-5>", lambda e: self.on_table_zoom(e, direction=-1))

        # Bottom-Right Frame: Graph
        graph_frame = ttk.LabelFrame(bottom_frame, text="Graph Visualization (Scroll to Zoom)")
        graph_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect('scroll_event', self.on_graph_zoom)

    # UI Helpers (Zoom & Toggles)
    def on_table_zoom(self, event, direction=None):
        if direction is None:
            direction = 1 if event.delta > 0 else -1
        if direction == 1:
            self.current_font_size += 1
        else:
            self.current_font_size = max(6, self.current_font_size - 1)
        self.update_table_style()
        return "break"

    def on_graph_zoom(self, event):
        if event.inaxes != self.ax: return
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()

    def toggle_inputs(self, event=None):
        method = self.method_var.get()
        if method == "Newton-Raphson":
            self.lbl_p1.config(text="Initial Guess (x0):")
            self.entry_p2.config(state="disabled")
            self.lbl_p2.config(state="disabled")
        elif method == "Secant":
            self.lbl_p1.config(text="First Guess (x0):")
            self.lbl_p2.config(text="Second Guess (x1):", state="normal")
            self.entry_p2.config(state="normal")
        else:
            self.lbl_p1.config(text="Lower Bound (xl):")
            self.lbl_p2.config(text="Upper Bound (xu):", state="normal")
            self.entry_p2.config(state="normal")

    # --- NEW: Helper for Dynamic Graphing ---
    def update_graph_view(self, event):
        """Callback to regenerate the function line when zooming/panning."""
        if self.current_f is None or self.main_line is None: 
            return
        
        # Get the new visible X-range
        xlim = self.ax.get_xlim()
        
        # Generate new X values to fill this range
        x_new = np.linspace(xlim[0], xlim[1], 1000)
        
        try:
            # Calculate new Y values using the stored function
            y_new = self.current_f(x_new)
            # Update the existing blue line with new data
            self.main_line.set_data(x_new, y_new)
        except:
            pass 

    # ---------------------------------------------------------
    # STEP 3: INTEGRATION (Connecting Logic to UI)
    # ---------------------------------------------------------
    def on_calculate(self):
        # Update Table: Clear previous results
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.ax.clear()

        try:
            # Get Data & Validation
            func_str = self.entry_func.get()
            method = self.method_var.get()
            p1 = float(self.entry_p1.get()) 
            tol = float(self.entry_tol.get())
            max_iter = int(self.entry_max_iter.get())

            # Store function for dynamic zooming
            f = lambda x: self.evaluate_function(func_str, x)
            self.current_f = f 

            if f(p1) is None: raise ValueError("Function invalid")

            # Computation: Call appropriate method
            root_val = None
            history = []

            if method == "Bisection":
                p2 = float(self.entry_p2.get())
                root_val, history = self.bisection(f, p1, p2, tol, max_iter)
            elif method == "False Position":
                p2 = float(self.entry_p2.get())
                root_val, history = self.false_position(f, p1, p2, tol, max_iter)
            elif method == "Newton-Raphson":
                root_val, history = self.newton_raphson(f, p1, tol, max_iter)
            elif method == "Secant":
                p2 = float(self.entry_p2.get())
                root_val, history = self.secant(f, p1, p2, tol, max_iter)

            if root_val is None and not history:
                messagebox.showerror("Math Error", "Method failed to converge or invalid bounds.")
                self.canvas.draw()
                return

            # Update Table: Populate with history list
            for row in history:
                self.tree.insert("", "end", values=row)

            # Update Graph: Plot f(x) and root
            margin = max(1, abs(root_val) * 0.5)
            x_vals = np.linspace(root_val - margin, root_val + margin, 1000)
            y_vals = f(x_vals)
            
            # --- NEW: Store line object for updates ---
            self.main_line, = self.ax.plot(x_vals, y_vals, label="f(x)", color="blue", linewidth=1.5)
            
            self.ax.axhline(0, color='black', linewidth=1) 
            self.ax.axvline(0, color='black', linewidth=1) 

            # --- NEW: Tangent Lines for Newton-Raphson ---
            if method == "Newton-Raphson":
                # Reconstruct sequence [Start, Iter1, Iter2...]
                x_sequence = [p1] + [float(row[3]) for row in history]
                
                # Draw tangents for first few iterations (limit to 10 to avoid clutter)
                for i in range(min(len(x_sequence) - 1, 10)): 
                    x_curr = x_sequence[i]
                    x_next = x_sequence[i+1] # The root approx (x-intercept)
                    y_curr = f(x_curr)
                    
                    # Draw Tangent (Red Dashed)
                    self.ax.plot([x_curr, x_next], [y_curr, 0], 
                                 color='red', linestyle='--', linewidth=1, alpha=0.6)
                    
                    # Draw vertical projection (Green Dotted)
                    if i < len(x_sequence) - 2:
                        y_next = f(x_next)
                        self.ax.plot([x_next, x_next], [0, y_next], 
                                     color='green', linestyle=':', alpha=0.3)

            # Standard Visualization for other methods
            else:
                approx_steps = [float(row[3]) for row in history]
                if len(approx_steps) > 1:
                    for step_x in approx_steps[:-1]:
                        step_y = f(step_x)
                        self.ax.plot([step_x, step_x], [0, step_y], color='green', linestyle=':', alpha=0.4)
                        self.ax.scatter([step_x], [0], color='green', s=20, alpha=0.5, label='_nolegend_')

            # Add scatter point at solution
            self.ax.scatter([root_val], [0], color="red", s=60, zorder=5, label=f"Root: {root_val:.5f}")
            
            # --- NEW: Connect dynamic zoom listener ---
            self.ax.callbacks.connect('xlim_changed', self.update_graph_view)

            # Redraw canvas
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend()
            self.ax.set_title(f"Method: {method} | Iterations: {len(history)} | Tol: {tol}")
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Input Error", f"Check your inputs:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RootFinderGUI(root)
    root.mainloop()