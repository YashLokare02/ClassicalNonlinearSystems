## Importing relevant libraries
import matplotlib.pyplot as plt

## For plotting the PDFs as a function of the number of optimization iterations
### Amend as necessary

## Example case (for a = 1, b = 2, \ell = 1/2; N = 4)
# Plotting the PDFs for other cases follows suit

## Plotting
fig, ax = plt.subplots(dpi = 600)

# Plot
ax.plot(x_diagonalization, y_diagonalization, 'k', linewidth = 2, label = 'Classical numerical diagonalization')

# Go for the bad solution
# Identify transitions between positive and negative y values
positive_mask = y_initial >= 0
negative_mask = y_initial < 0
transitions = np.where(np.diff(positive_mask.astype(int)) != 0)[0]

# Include start and end indices
segments = np.concatenate(([0], transitions + 1, [len(y_initial)]))

# Flags to ensure only one label per type of line
solid_line_plotted = False
dotted_line_plotted = False

# Plot each segment separately
for i in range(len(segments) - 1):
    start, end = segments[i], segments[i + 1]
    if positive_mask[start]:
        if not solid_line_plotted:
            plt.plot(x_initial[start:end], np.abs(y_initial[start:end]), linestyle='-', color='purple', \
                     linewidth = 2, label = "Initial ansatz")
            solid_line_plotted = True
        else:
            plt.plot(x_initial[start:end], np.abs(y_initial[start:end]), linestyle='-', color='purple', linewidth = 2)
    else:
        if not dotted_line_plotted:
            plt.plot(x_initial[start:end], np.abs(y_initial[start:end]), linestyle=':', color='purple', linewidth = 2)
            dotted_line_plotted = True
        else:
            plt.plot(x_initial[start:end], np.abs(y_initial[start:end]), linestyle=':', color='purple', linewidth = 2)

# Continue plotting
# ax.plot(x_vqe_noiseless_6000, y_vqe_noiseless_6000, 'b', linewidth = 2, alpha = 0.7, label = 'Ideal simulator (poor initial ansatz [6000 iterations])')
# ax.plot(x_vqe_noiseless_12000, y_vqe_noiseless_12000, 'y', linewidth = 2, alpha = 0.5, label = 'Ideal simulator (poor initial ansatz [12000 iterations])')
ax.plot(x_vqe_noisy_150, y_vqe_noisy_150, 'b', linewidth = 2, label = 'Manila machine (150 iterations)')
ax.plot(x_vqe_noisy_300, y_vqe_noisy_300, color = 'r', linewidth = 2, label = 'Manila machine (300 iterations)')
ax.plot(x_vqe_noisy_700, y_vqe_noisy_700, color = 'saddlebrown', linewidth = 2, label = 'Manila machine (700 iterations)')

# Set labels and title
ax.set_xlabel('$x$', fontsize = 17)
ax.set_ylabel('$P(x)$', fontsize = 17)
ax.set_title(r'$a = 1, b = 2, \ell = 1/2 ~(7000 ~\mathrm{shots})$; $N = 2$', fontsize = 13)

ax.grid(True)

ax.set_yscale('log')

ax.set_ylim(2 * 1e-3, 1)

# Show a legend
legend = ax.legend(loc = 'upper left', bbox_to_anchor=(-0.05, 1.26), ncol=2, prop = {'size': 9})

## Plotting relative errors and <H> via the quantum-computed zeromodes for the noiseless and noisy cases

# Note: Use the quantum-computed zeromodes to compute <x^2>, relative errors in <x^2>, and <H>

# Plot
fig, ax1 = plt.subplots(dpi = 600)

ax1.set_xlabel('Iterations', fontsize = 17)
ax1.set_ylabel('Relative error', fontsize = 17)
ax1.plot(iterations, errors_descent, color = 'blue', marker = 's', linestyle = '-', markersize = 5, linewidth = 2.5, label='Relative error (simulation)')
ax1.plot(iterations, errors_hardware, color = 'red', marker = 'o', linestyle = '-', markersize = 5, linewidth = 2.5, label='Relative error (Manila machine)')
# ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_yscale('log')
ax1.grid(True)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel(r'$\langle H \rangle$', fontsize = 17)  # we already handled the x-label with ax1
ax2.plot(iterations, expectations, color = 'blue', marker = 's', linestyle = '--', markersize = 5, label=r'$\langle H \rangle$ (simulation)')
ax2.plot(iterations, expectation_hardware, color = 'red', marker = 'o', linestyle = '--', markersize = 5, label=r'$\langle H \rangle$ (Manila machine)')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_yscale('log')
ax2.grid(True)
fig.suptitle(r'$a = 1, b = 2, \ell = 1/2; N = 4$', fontsize = 15)

# Add a legend
# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', bbox_to_anchor=(1.005, 1), ncol = 2)

# Save the plot
plt.tight_layout(pad=1.0)
plt.savefig("Positive_a_N_4_relative_error_expectation_IMFIL_hardware_simulation.pdf")

## Plotting the VQE convergence plot

# Here, 'sol' corresponds to the energy eigenvalue obtained using NumpyMinimumEigensolver

plt.rcParams["font.size"] = 14
plt.figure(dpi = 600)
plt.figure(figsize=(12, 6))
plt.plot(log.values, label="VQE")
plt.axhline(y=sol.eigenvalue, color="tab:red", ls="--", label="Target")
plt.legend(loc="best")
plt.xlabel("Iteration")
plt.ylabel(r'$\langle H \rangle$')
plt.title("VQE convergence")
plt.show()