from .simulator import *
import matplotlib.pyplot as plt
import os
from icecream import ic
import matplotlib
#matplotlib.use('AGG')
import subprocess
import os
from icecream import ic
from .psiviz import phase_mag_vis2, mag_vis, dens_vis, phase_cmap, dens_cmap
import colorcet
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties



class AnimatorBase:
    """Base class for animators.

    This class is not meant to be used directly. Instead, it should be subclassed.
    It defines the interface for animators. The subclass should implement the
    following methods:
    - init_figure: initialize the figure before any frames are made
    - make_frame: make a frame for the animation by updating the figure. Should call super().make_frame() first.

    Attributes:
        simulator: The simulator object used for the animation.
        folder: The folder path where the frames will be saved.
        fig_width: The width of the figure in inches.
        width: The width of the frame in pixels.
        height: The height of the frame in pixels.
        fig_height: The height of the figure in inches.
        dpi: The dots per inch of the figure.
        interval: The animation time step skip interval.
        basename: The base name of the frame files.
        digits: The number of digits used in the frame file names.
        extension: The file extension of the frame files.
        format: The format string for the frame file names.
        frame_postprocess: A function to be called after each frame is made.
        frame_index: The index of the current frame.
        frame_list: A list of frame file names.

    """

    def __init__(self, simulator, folder = './frames/'):
        self.simulator = simulator

        self.folder = folder
        # make sure folder exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.fig_width = 10
        self.set_framesize(800, 600)
        self.skip_interval = 1
        self.basename = 'frame'
        self.digits = 5
        self.extension = '.png'
        self.format = self.folder + self.basename + '%0' + str(self.digits) + 'd' + self.extension
        ic(self.format)
        
        # set up caption
        self.caption_format = 't = {sim.t:.2f}'
        self.caption_pos = (0.05, 0.95) # position of caption in figure
        self.caption_font = None # custom font for caption
        
        
        # set up axis.
        # it is the responsibility of the subclass
        # to properly set up the axis
        self.show_axis = True
       
    def set_style(self, style):
        """ Set the style of the animation. """
        self.style = style
        
        style.set_anim(self)

        self.set_frame_postprocess(
            lambda anim: style.frame_postprocess(anim)
        )
        
    def get_caption(self):
        """
        Format the caption text. To be run by subclass within make_frame()
        """
        return self.caption_format.format(sim=self.simulator)
    
    def place_caption(self):
        """ Place the caption in the figure. To be run by subclass within init_figure(). """
        
        # set caption
        caption_text = self.get_caption()
        ic(self.caption_font)
        # z index should be frontmost
        self.caption = plt.text(
            *self.caption_pos, 
            caption_text, 
            transform=self.fig.transFigure, 
            ha='left', 
            fontproperties=self.caption_font,
            zorder=100
        )
        
    

    def set_framesize(self, width_pixels, height_pixels):
            """Set the frame size for the animation.

            Args:
                width_pixels (int): The width of the frame in pixels.
                height_pixels (int): The height of the frame in pixels.

            Returns:
                None
            """
            self.fig_width_pixels = width_pixels
            self.fig_height_pixels = height_pixels
            self.fig_height = self.fig_height_pixels / self.fig_width_pixels * self.fig_width
            self.dpi = self.fig_width_pixels / self.fig_width

    def set_interval(self, skip_interval):
        """Set the animation time step skip interval.

        Args:
            skip_interval: The interval between frames.

        """
        self.skip_interval = skip_interval

    def init_figure(self):
        """
        Initialize the figure before any frames are made.
        
        This method initializes the figure by performing the following steps:
        1. Saves the current matplotlib backend.
        2. Sets the backend to 'AGG'.
        3. Creates a new figure and axes using the specified figure width and height.
        4. Adjusts the subplot parameters to fit the figure within the specified boundaries.
        """
        
        # get the current matplotlib backend and save it
        self.previous_backend = matplotlib.get_backend()
        # set new backend
        matplotlib.use('AGG')

        if hasattr(self, 'style'):
            self.style.mpl_style()
            
        # make figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)



            
    def set_frame_postprocess(self, frame_postprocess):
        """Set a function to be called after each frame is made.

        Args:
            frame_postprocess: A function to be called after each frame is made.

        """
        self.frame_postprocess = frame_postprocess
        
        

    def get_callback(self):
        """Returns a callback function for the simulator.

        Returns:
            A callback function that can be used with the simulator.

        """
        # initialize figure if not already done
        if not hasattr(self, 'fig'):
            # run init_figure from subclass
            self.init_figure()

        # set up frame list
        self.frame_index = 0
        self.frame_list = []

        def callback(simulator):
            if simulator.t_index % self.skip_interval == 0:
                
                self.make_frame()
                self.save_frame()
                if hasattr(self, 'frame_postprocess'):
                    self.frame_postprocess(self)

        return callback

    def make_frame(self):
        """Make a frame for the animation.

        This method should be implemented by the subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        """
        raise NotImplementedError()

    def save_frame(self):
        """Save the current frame.

        This method is called by the callback function, see get_callback().

        """
        # Generate the filename for the current frame
        filename = self.format % self.frame_index
        
        # Save the figure as an image file with the generated filename
        self.fig.savefig(filename, dpi=self.dpi)

        # Increment the frame index and add the filename to the frame list
        self.frame_index += 1
        self.frame_list.append(filename)

    def get_frame(self, index):
        """Get a frame from the animation, read from disk. Useful if you want to display a single frame in a notebook 
        or script.

        Args:
            index: The index of the frame to get.

        Returns:
            The frame image as a numpy array.

        """
        filename = self.frame_list[index]
        return plt.imread(filename)
    

    def make_movie(self, filename):
        """Make a movie from the frames.

        Args:
            filename: The name of the output movie file.

        """
        ic(len(self.frame_list))

        command = ('ffmpeg',
                   '-r', '24',
                   '-i', self.format,
                   '-r', '24',
                   '-pix_fmt', 'yuv420p',
                   '-vframes', str(len(self.frame_list)),
                   '-y',  # Overwrite existing file
                   filename)

        subprocess.call(command)
        
    def clean_frames(self):
        """Remove the frame files."""
        for filename in self.frame_list:
            os.remove(filename)
            
    def restore_backend(self):
        """Restore the previous matplotlib backend."""
        matplotlib.use(self.previous_backend)
        
            



class Animator1d(AnimatorBase):
    """
    Animator1d class for creating 1D animations.

    This class is a subclass of AnimatorBase and implements the necessary methods
    for creating 1D animations. It initializes the figure, sets up the plot, and
    updates the plot for each frame of the animation.

    Attributes:
        simulator: The simulator object used for the animation.

    Methods:
        __init__(self, simulator): Initializes the Animator1d object.
        init_figure(self): Initializes the figure and sets up the plot.
        make_frame(self): Updates the plot for each frame of the animation.
    """

    def __init__(self, simulator, folder = './frames/'):
        """
        Initialize the Animator1d object.

        Args:
            simulator: The simulator object used for the animation.
        """
        super().__init__(simulator, folder = folder)
        
        
        self.real_color = 'C0'
        self.imag_color = 'C1'
        self.abs_color = 'C2'
        self.pot_color = 'C3'
        self.xtick_pad = 15
        self.show_legend = False
        
        self.xlim = None # set to [x0, x1] to zoom in on a region

    
    def get_potential_curve(self):
        """ Get potential curve for plotting potential. 
        
        This function appropriately scales the potential plot so that it is
        visible in the animation.
        
        The computation of the max and min values of the potential is done
        based on the time-independent part of the potential, and the laser
        pulse is added to the potential plot.
        
        Called internally by init_figure() and make_frame().
        
        """
        
        x = self.simulator.x
        psi = self.simulator.psi
        t = self.simulator.t
        Vmax = np.max(self.simulator.ham.V)
        Vmin = np.min(self.simulator.ham.V)
        D = self.simulator.laser_pulse_fun(t) * self.simulator.ham.D
        Vplot = (self.simulator.ham.V + D - Vmin) / (Vmax - Vmin)

        return Vplot * 0.5 - 0.75
        
        
    def init_figure(self):
        """
        Initialize the figure and set up the plot.

        This method plots the wavefunction,
        potential energy, and sets the plot limits and tick parameters.
        """

        super().init_figure()

        x = self.simulator.x
        psi = self.simulator.psi
        # Vmax = np.max(self.simulator.ham.V)
        # Vmin = np.min(self.simulator.ham.V)
        # ic(Vmax, Vmin)
        # Vplot = (self.simulator.ham.V - Vmin) / (Vmax - Vmin)

        # plot wavefunction
        self.rline = plt.plot(x, psi.real, self.real_color)
        self.iline = plt.plot(x, psi.imag, self.imag_color)
        self.aline = plt.plot(x, np.abs(psi), self.abs_color)
        self.potline = plt.plot(x, self.get_potential_curve(), self.pot_color)

        # add legend
        labels = ['real', 'imag', 'abs', 'pot']
        if self.show_legend:
            self.ax.legend(labels, loc='upper right', prop={'size': 16})


        # zoom in if xlim is set
        if self.xlim is not None:
            plt.xlim(self.xlim[0], self.xlim[1])

        plt.ylim(-1, 1)

        # set caption
        self.place_caption()
        
        
        # draw axis
        if self.show_axis:
            ic('show axis')
            # turn axis labels inwards
            #self.ax.tick_params(axis="y", direction="in", pad=-22)
            self.ax.tick_params(axis="x", direction="in", pad=-self.xtick_pad)
            # turn off vertical tick labels
            self.ax.set_yticklabels([])
            # turn off vertical tick marks
            self.ax.tick_params(axis='y', which='both', length=0)
            # turn off visibility for leftmost tick on x axis
            self.ax.get_xticklabels()[0].set_visible(False)
            self.ax.get_xticklabels()[-1].set_visible(False)
            # turn off visibility for leftmost tick mark on x axis
            self.ax.get_xticklines()[0].set_visible(False)
            self.ax.get_xticklines()[-1].set_visible(False)
        else:
            self.ax.axis('off')            

        # turn off spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)




    def make_frame(self):
        """
        Update the plot for each frame of the animation.

        This method updates the wavefunction plot, potential energy plot, and the caption
        text for each frame of the animation.
        """

        temp = self.simulator.psi
        self.rline[0].set_ydata(temp.real)
        self.iline[0].set_ydata(temp.imag)
        self.aline[0].set_ydata(np.abs(temp))
        self.potline[0].set_ydata(self.get_potential_curve())
        
        caption_text = self.get_caption()
        self.caption.set_text(caption_text)
        



    
    
    


class Animator2d(AnimatorBase):
    """
    Animator1d class for creating 2D animations.

    This class is a subclass of AnimatorBase and implements the necessary methods
    for creating 2D animations. It initializes the figure, sets up the plot, and
    updates the plot for each frame of the animation.

    Attributes:
        simulator: The simulator object used for the animation.

    Methods:
        __init__(self, simulator): Initializes the Animator1d object.
        init_figure(self): Initializes the figure and sets up the plot.
        make_frame(self): Updates the plot for each frame of the animation.
    """

    def __init__(self, simulator, folder = './frames/'):
        """
        Initialize the Animator1d object.

        Args:
            simulator: The simulator object used for the animation.
        """
        super().__init__(simulator, folder = folder)
        
        self.xlim = None # set to [x0, x1] to zoom in on a region
        self.ylim = None # set to [y0, y1] to zoom in on a region
        
        self.vis_type = 'complex'
        self.energy_shift = 0.0 # to shift frequency of phase oscillations. premultiplies wavefunction with exp(1j * energy_shift * t) before visualization.
        #self.vis_type = 'magnitude'
        self.mag_map = lambda x: x
        self.dens_factor = 0.75 # fraction of max density to show
        self.phase_cmap = colorcet.cm['CET_C6']
        #self.phase_cmap = colorcet.cm['CET_C3s']
        self.mag_cmap = colorcet.cm['CET_L2']
        


    def get_extent(self):
        """ Get extent of image plot. Used internally. """
        
        x_range = self.simulator.x
        y_range = self.simulator.y
        return [np.min(x_range), np.max(x_range), np.min(y_range), np.max(x_range)]

    def init_figure(self):
        """
        Initialize the figure and set up the plot.

        """

        
#        fig_width = self.fig_width
#        fig_height = self.fig_height
        
#        self.fig, self.ax = plt.subplots(1, 1, figsize = (fig_width,fig_height))
#        plt.axis('off')
#        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


        super().init_figure()
                

        psi = self.simulator.psi
        

        # make RGB bitmap            
        self.make_bmp(psi) # self.bmp

        # plot bitmap
        self.ax.clear()
        self.image = self.ax.imshow(
            self.bmp,
            origin='lower', 
            extent = self.get_extent()
        )
        
        # set xlim and ylim
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim[0], self.xlim[1])
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim[0], self.ylim[1])
            
        # set caption
        self.place_caption()    
        
        
        if self.show_axis:
            # move tick labels to the inside 
            self.ax.tick_params(axis="y", direction="in", pad=-22)
            self.ax.tick_params(axis="x", direction="in", pad=-15)        
        else:
            self.ax.axis('off')


    def add_potential_visualization_2d(self, transparent_range=[0, 0.1], cmap=dens_cmap):
        # customize figure by adding potential visualization
        if self.simulator.dim != 2:
            raise ValueError("add_potential_visualization_2d only works for 2d simulations")
        
        V = self.simulator.ham.V
        # add potential visualization. use imshow, and make values below 0.1 transparent
        masked_V = np.ma.masked_where((V < transparent_range[1]) & (V >= transparent_range[0]), V)
        norm = matplotlib.colors.Normalize(vmin=transparent_range[1], vmax=V.max())
        cmap.set_bad(color='none') 
        self.im_pot = self.ax.imshow(
            masked_V.T, 
            cmap=cmap, 
            norm=norm, 
            extent=self.get_extent(),
            origin='lower'
        )


    def make_bmp(self, psi):
        """
        Compute bitmap visualization of the 2d wavefunction in the simulator.
        """

        dens_factor = self.dens_factor
    
        # compute max density, if not already computed
        # that is, if we are at the first frame ... 
        if not hasattr(self.simulator, 'max_dens'):
            max_dens = np.abs(psi).max() * dens_factor
            scale = np.abs(psi).max() * dens_factor
        

        phase_factor = np.exp(1j * self.energy_shift * self.simulator.t)
        if self.vis_type == 'complex':
            bmp = phase_mag_vis2(phase_factor * psi.T / scale, cmap=self.phase_cmap, mag_map=self.mag_map)
        elif self.vis_type == 'magnitude':    
            bmp = mag_vis(psi.T/scale, cmap=self.mag_cmap,  mag_map=self.mag_map)
        else:    
            raise ValueError(f"Unknown vis_type: {self.vis_type}")

        self.bmp = bmp
        

    def make_frame(self):
        """
        Update the plot for each frame of the animation.


        """
        

        self.make_bmp(self.simulator.psi)
        self.image.set_data(self.bmp)
        
        caption = self.caption_format.format(sim=self.simulator)
        self.caption.set_text(caption)

    
    




class Style:
    """
    A class representing the style of an animation.

    Attributes:
        anim: The animation object associated with the style.

    Methods:
        set_anim: Sets the animation object for the style.
        mpl_style: Applies the Matplotlib style to the animation.
        frame_postprocess: Performs post-processing on each frame of the animation.
    """

    def __init__(self):
        pass
    
    def set_anim(self, anim):
        self.anim = anim

    def mpl_style(self):
        pass
    
    def frame_postprocess(self):
        pass
    
    
class DarkTheme(Style):

    def mpl_style(self):
        # set dark theme in matplotlib

        ic('inside mpl_style')
        
        plt.style.use('dark_background')
        
        
        colors = ['#00ff9f', '#d600ff', '#feff6e', '#001eff' ,'#00b8ff']
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
        
        # set thicker lines for plots
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['axes.linewidth'] = 2
        
        # set font size for labels
        mpl.rcParams['axes.labelsize'] = 20
        
        # set font to be Manrope for all text
#        mpl.rcParams['font.family'] = 'JetBrains Mono'
        # Set alternative font if JetBrains Mono is not available,
        # include system default in list of sans-serif fonts
        mpl.rcParams['font.sans-serif'] = ['JetBrains Mono', 'DejaVu Sans']
        
        # set font size for ticks
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        # set tick marker size
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['xtick.major.width'] = 1

        
        # set caption location in anim
        self.anim.caption_pos = (0.06, 0.9)
        self.anim.caption_font = FontProperties(size=20, style='italic')
        self.anim.xtick_pad = 25 
        
        if isinstance(self.anim, Animator1d):
            # add legend to anim.ax, labels are 'real', 'imat', 'abs', 'pot'
            self.anim.show_legend = True    
        
        
    
    def frame_postprocess(self, anim : AnimatorBase):
        
        return None
        