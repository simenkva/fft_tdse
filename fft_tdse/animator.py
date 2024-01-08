from .simulator import *
import matplotlib.pyplot as plt
import os
import shutil
from icecream import ic
import matplotlib
import subprocess
import os
from icecream import ic
from .psiviz import phase_mag_vis2, mag_vis, dens_vis, phase_cmap, dens_cmap
import colorcet
from IPython.display import clear_output, display, update_display
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import mpl_toolkits.axisartist as axisartist
from PIL import Image
from .is_notebook import is_notebook





class AnimatorBase:
    """Base class for animators.

    This class is not meant to be used directly. Instead, it should be subclassed.
    It defines the interface for animators. The subclass should implement the
    following methods:
    - init_figure: initialize the figure before any frames are made
    - make_frame: make a frame for the animation by updating the figure. Should call super().make_frame() first.

    Attributes:
        name: The name of the animation. Overrides folder if exists.
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

    def __init__(self, simulator, name = 'animation'):
        """Initialize the AnimatorBase object.

        It is advisable to set a name for the animation, since frames will be saved in a folder with the name of the animation.
        Using a unique name for each animation will prevent frames from different animations from being mixed up,
        and will allow the animations to be run in parallel.

        Args:
            simulator (Simulator): The simulator object used for the animation.
            name (str): The name of the animation. Used for output file naming, including the frames folder.
        """
        
        # save Simulator instance
        self.simulator = simulator

        # set up name
        self.name = name
        ic(self.name)
        
        # set up folder for storing frames
        self.folder = f'./{self.name}_frames/'
        ic(self.folder)
        
        # make sure folder exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


        # set up figure width in inches. height is computed from width and aspect ratio
        self.fig_width = 10
        # set up frame size in pixels.
        self.set_framesize(800, 600)
        # set up interval between frames
        self.skip_interval = 1
        # set up frame file name format
        self.basename = 'frame'
        self.digits = 6
        self.extension = '.png'
        self.format = self.folder + self.basename + '%0' + str(self.digits) + 'd' + self.extension
        ic(self.format)
        
        # set up preview inline in notebook settings
        self.preview = False # default = no preview
        self.preview_interval = 1 # interval between frames in preview. default value implies every frame is shown.
        
        # set up caption
        self.caption_format = 't = {sim.t:.2f}'
        self.caption_pos = (0.05, 0.95) # position of caption in figure
        self.caption_font = None # custom font for caption
        
        
        # set up axis.
        # it is the responsibility of the subclass
        # to properly set up the axis
        self.show_axis = True
        
        # add inset axes params
        self.axins_shape = [0.8, 0.8, 0.18, 0.18]
        self.axins_dot_size = 10

       
    def set_style(self, style):
        """ Set the style of the animation. 
        
        At the moment, the only style programmed is DarkTheme():
        ```python
        anim.set_style(DarkTheme())
        ```
        
        Args:
            style: A Style object.
        
        """


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
#            transform=self.fig.transFigure, 
            transform=self.ax.transAxes,
            ha='left', 
            fontproperties=self.caption_font,
            zorder=100
        )
        
    
    def set_figwidth(self, fig_width):
        """Set the width of the figure.

        Args:
            fig_width: The width of the figure in inches. The height is computed from the aspect ratio if the figure size in pixels.

        """
        self.fig_width = fig_width
        self.fig_height = self.fig_height_pixels / self.fig_width_pixels * self.fig_width
        self.dpi = self.fig_width_pixels / self.fig_width
        
    def set_framesize(self, width_pixels, height_pixels):
            """Set the frame size for the animation, in pixels.

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
            
            ic(self.fig_width_pixels, self.fig_height_pixels, self.fig_width, self.fig_height, self.dpi)


    def set_interval(self, skip_interval):
        """Set the animation time step skip interval.

        Args:
            skip_interval: The interval between frames.

        """
        self.skip_interval = skip_interval

    def set_preview(self, preview, preview_interval = 1):
        """Set whether to show a preview of the animation inline in a notebook.

        Args:
            preview: True to show a preview, False otherwise.

        """
        self.preview = preview
        self.preview_interval = preview_interval
            
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
        self.fig.tight_layout()

        if self.show_axis:
            pass
            # other settings ... ?
            
        else:
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0) #was plt.subplots_adjust
            self.ax.axis('off')
            

    def add_laser_visualization(self):
        """ Add inset axes for visualizing the laser pulse. """
        self.axins = self.ax.inset_axes(self.axins_shape)
        # reduce tick label font size
        #self.axins.tick_params(axis='both', which='major', labelsize=12)
        
        # remove ticks on both axes
        self.axins.set_xticks([])
        self.axins.set_yticks([])
        
        self.axins.plot(
            self.simulator.t_grid, 
            self.simulator.laser_pulse_fun(self.simulator.t_grid), 
            color='white'
        )
        self.axins_dot =  self.axins.plot(
            self.simulator.t, 
            self.simulator.laser_value, 
            color='white', 
            marker='o',
            markersize=self.axins_dot_size,
            markeredgecolor='white'
        )
        

        # Increase limits slightly, for example by 10%
        ymin, ymax = self.axins.get_ylim()
        ymin_new = ymin - 0.1 * (ymax - ymin)
        ymax_new = ymax + 0.1 * (ymax - ymin)

        # Set new limits
        self.axins.set_ylim(ymin_new, ymax_new)    
                
        


            
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
                self.update_laser_visualization()
                self.save_frame()
                if hasattr(self, 'frame_postprocess'):
                    self.frame_postprocess(self)
                    
                # preview in notebook
                i = len(self.frame_list) - 1
                if i % self.preview_interval == 0:
                    if self.preview and is_notebook():
                        # display frame in notebook
                        f = self.get_frame(-1)
                        if i == 0:
                            display(f, display_id=self.name)
                        else:
                            update_display(f, display_id=self.name)

        return callback


    def update_laser_visualization(self):
        """ Update the inset laser visualization plot"""
        if hasattr(self, 'axins'):
            # change x and y position of dot
            self.axins_dot[0].set_xdata(self.simulator.t)
            self.axins_dot[0].set_ydata(self.simulator.laser_value)

        
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
        
    def copy_frame(self, index):
        """ Coppy a frame and advance counter by one. """
        
        # Generate the filename for the current frame
        filename = self.format % self.frame_index
        
        # copy the file pointed to by index
        shutil.copyfile(self.frame_list[index], filename)
        
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
        #return plt.imread(filename)
        return Image.open(filename)
    
    

    def make_movie(self, filename = None, duplicate_last_frame = 0):
        """Make a movie from the frames.

        Args:
            filename: The name of the output movie file.

        """
        ic(len(self.frame_list))
        
        # # this is a hack for now. first frame is
        # # duplicated in the movie. so we remove it.
        # # copy frame list
        # frame_list = self.frame_list.copy()
        # # pop first frame
        # frame_list.pop(0)
        
        # add last frame a number of times
        # since ffmpeg seems to drop a few frames
        # in certain cases
        if duplicate_last_frame > 0:
            for i in range(duplicate_last_frame):
                self.copy_frame(-1)

        if filename is None:
            filename = self.name + '.mp4'

        ic('making movie ...')
        ic(filename)

        command = ('ffmpeg',
                   '-r', '24',
                   '-i', self.format,
                   '-r', '24',
                   '-pix_fmt', 'yuv420p',
                   '-vframes', str(len(self.frame_list)),
                   '-y',  # Overwrite existing file
                   filename)


        # store exit code
        exit_code = subprocess.call(command)
        ic(exit_code)

        
    def clean_frames(self):
        """Remove the frame files."""
        for filename in self.frame_list:
            os.remove(filename)
        # delete folder
        os.rmdir(self.folder)
        # close figure
        plt.close(self.fig)
        self.restore_backend()
        
            
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

    def __init__(self, simulator, name = None):
        """
        Initialize the Animator1d object.

        Args:
            simulator: The simulator object used for the animation.
        """
        super().__init__(simulator, name = name)
        
        
        self.real_color = 'C0'
        self.imag_color = 'C1'
        self.abs_color = 'C2'
        self.pot_color = 'C3'
        self.xtick_pad = 15
        self.show_legend = False
        self.semilogy = False
        
        self.xlim = [self.simulator.grid.a[0], self.simulator.b[0]] #None # set to [x0, x1] to zoom in on a region
        self.ylim = [-1.0, 1.0]
    
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
        
    def add_legend(self, loc = 'upper right'):
        """ Add legend to anim.ax, labels are 'real', 'imat', 'abs', 'pot' """
        self.labels = ['real', 'imag', 'abs', 'pot']
        if self.show_legend:
            self.ax.legend(self.labels, loc=loc, prop={'size': 16})

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
        
        if self.semilogy == True:
            self.rline = plt.semilogy(x, np.abs(psi.real), self.real_color)
            self.iline = plt.semilogy(x, np.abs(psi.imag), self.imag_color)
            self.aline = plt.semilogy(x, np.abs(psi), self.abs_color)
            
            
        else:
                
            self.rline = plt.plot(x, psi.real, self.real_color)
            self.iline = plt.plot(x, psi.imag, self.imag_color)
            self.aline = plt.plot(x, np.abs(psi), self.abs_color)

        # create new axis on top of current axis, but with linear scale
        self.ax2 = self.ax.twinx()
        self.ax2.set_yscale('linear')
        self.ax2.set_ylim(-1,1)
        self.potline = self.ax2.plot(x, self.get_potential_curve(), self.pot_color)


        # add legend
        self.add_legend(loc = 'upper right')


        # zoom in if xlim is set
        if self.xlim is not None:
            plt.xlim(self.xlim[0], self.xlim[1])

        self.ax.set_ylim(self.ylim[0], self.ylim[1])

        # set caption
        self.place_caption()
        
        
        # draw axis
        if self.show_axis:
            ic('show axis 1d')
            # turn axis labels inwards
            #self.ax.tick_params(axis="y", direction="in", pad=-22)
            #self.ax.tick_params(axis="x", direction="in", pad=-self.xtick_pad)
            # turn off vertical tick labels
            self.ax.set_yticklabels([])
            self.ax2.set_yticklabels([])
            # turn off vertical tick marks
            self.ax.tick_params(axis='y', which='both', length=0)
            self.ax2.tick_params(axis='y', which='both', length=0)
            # turn off visibility for leftmost tick on x axis
        #     self.ax.get_xticklabels()[0].set_visible(False)
        #     self.ax.get_xticklabels()[-1].set_visible(False)
        #     # turn off visibility for leftmost tick mark on x axis
        #     self.ax.get_xticklines()[0].set_visible(False)
        #     self.ax.get_xticklines()[-1].set_visible(False)
            # turn off vertical tick labels and tick marks
            
        else:
            self.ax.axis('off')          
            self.ax2.axis('off')

        # # turn off spines
        # self.ax.spines['top'].set_visible(False)
        # self.ax.spines['right'].set_visible(False)
        # self.ax.spines['bottom'].set_visible(False)
        # self.ax.spines['left'].set_visible(False)




    def make_frame(self):
        """
        Update the plot for each frame of the animation.

        This method updates the wavefunction plot, potential energy plot, and the caption
        text for each frame of the animation.
        """

        temp = self.simulator.psi

        if not self.semilogy:
            self.rline[0].set_ydata(temp.real)
            self.iline[0].set_ydata(temp.imag)
            self.aline[0].set_ydata(np.abs(temp))
            self.potline[0].set_ydata(self.get_potential_curve())
        else:
            self.rline[0].set_ydata(np.abs(temp.real))
            self.iline[0].set_ydata(np.abs(temp.imag))
            self.aline[0].set_ydata(np.abs(temp))
            self.potline[0].set_ydata(self.get_potential_curve())
            
        # self.rline[0].set_ydata(temp.real)
        # self.iline[0].set_ydata(temp.imag)
        # self.aline[0].set_ydata(np.abs(temp))
        # self.potline[0].set_ydata(self.get_potential_curve())
        
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

    def __init__(self, simulator, name = None):
        """
        Initialize the Animator1d object.

        Args:
            simulator: The simulator object used for the animation.
        """
        super().__init__(simulator, name = name)
        
        self.xlim = None # set to [x0, x1] to zoom in on a region
        self.ylim = None # set to [y0, y1] to zoom in on a region
        
        self.no_scale = False # set to True to turn off density scaling
        self.vis_type = 'complex'
        self.energy_shift = 0.0 # to shift frequency of phase oscillations. premultiplies wavefunction with exp(1j * energy_shift * t) before visualization.
        #self.vis_type = 'magnitude'
        self.mag_map = lambda x: x
        self.mag_enhance = None # set to a scalar or a grid function to scale magnitude
        self.dens_factor = 0.75 # scale density before visualization, so that larger values can be shown
        self.phase_cmap = colorcet.cm['CET_C6']
        #self.phase_cmap = colorcet.cm['CET_C3s']
        self.mag_cmap = colorcet.cm['CET_L16']
        

        


    def get_extent(self):
        """ Get extent of image plot. Used internally. """
        
        x_range = self.simulator.grid.x[0]
        y_range = self.simulator.grid.x[1]
        ic()
        ic(x_range.shape, y_range.shape)
        
        extent = [np.min(x_range), np.max(x_range), np.min(y_range), np.max(y_range)]
        ic(extent)
        return extent

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
        
        self.image = self.ax.imshow(
            self.bmp,
            origin='lower', 
            aspect='equal',
            extent = self.get_extent()
        )
        #self.ax.autoscale(False)
        self.ax.set_aspect('equal')
        
       
        # set xlim and ylim
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim[0], self.xlim[1])
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim[0], self.ylim[1])
        
        
        
        # set caption
        self.place_caption()    
        
        
        if self.show_axis:
            # # move tick labels to the inside 
            # #self.ax.tick_params(axis="y", direction="in", pad=-15)
            # #self.ax.tick_params(axis="x", direction="in", pad=-25)
            # # left align y tick labels
            # #self.ax.axis["left"].major_ticklabels.set_ha("left")
            # # turn off visibility for leftmost tick on x axis
            # self.ax.get_xticklabels()[0].set_visible(False)
            # self.ax.get_xticklabels()[-1].set_visible(False)
            # # turn off visibility for leftmost tick mark on x axis
            # self.ax.get_xticklines()[0].set_visible(False)
            # self.ax.get_xticklines()[-1].set_visible(False)                
            # # turn off visibility for leftmost tick on y axis
            # self.ax.get_yticklabels()[0].set_visible(False)
            # self.ax.get_yticklabels()[-1].set_visible(False)
            # # turn off visibility for leftmost tick mark on y axis
            # self.ax.get_yticklines()[0].set_visible(False)
            # self.ax.get_yticklines()[-1].set_visible(False)
            
            pass
        else:
            self.ax.axis('off')


    def add_potential_visualization_2d(self, transparent_range=[0, 0.1], cmap=dens_cmap, contour=False):
        """ Add potential visualization to 2d animation.
        
        Args:
            transparent_range: range of potential values to make transparent
            cmap: colormap to use for potential visualization
            contour: whether to use contour plot instead of imshow
        
        
        Returns:
            None
        """
        
        
        # customize figure by adding potential visualization
        if self.simulator.dim != 2:
            raise ValueError("add_potential_visualization_2d only works for 2d simulations")
        
        V = self.simulator.ham.V
        # add potential visualization. use imshow, and make values below 0.1 transparent
        masked_V = np.ma.masked_where((V < transparent_range[1]) & (V >= transparent_range[0]), V)
        norm = matplotlib.colors.Normalize(vmin=transparent_range[1], vmax=V.max())
        cmap.set_bad(color='none') 

        if contour:

            self.im_pot = self.ax.contour(
                V.T, 
                cmap=cmap,
                extent=self.get_extent(),
                linewidths=plt.rcParams['lines.linewidth']/2,
                alpha=0.35
            )
    
        else:        
            self.im_pot = self.ax.imshow(
                masked_V.T, 
                cmap=cmap, 
                norm=norm, 
                extent=self.get_extent(),
                origin='lower',
                aspect='equal'
            )
            
            


    def make_bmp(self, psi):
        """
        Compute bitmap visualization of the 2d wavefunction in the simulator.
        """

        dens_factor = self.dens_factor
    
        # compute max density, if not already computed
        # that is, if we are at the first frame ... 
        if not hasattr(self, 'max_dens'):
            self.max_dens = np.abs(psi).max() * dens_factor
        scale = self.max_dens * dens_factor
        
        # hack to turn of density scaling
        if self.no_scale:
            scale = 1.0

        if self.mag_enhance is not None:
            psi_for_vis = self.mag_enhance * psi
        else:
            psi_for_vis = psi
            

        phase_factor = np.exp(1j * self.energy_shift * self.simulator.t)
        if self.vis_type == 'complex':
            bmp = phase_mag_vis2(phase_factor * psi_for_vis.T / scale, cmap=self.phase_cmap, mag_map=self.mag_map)
        elif self.vis_type == 'magnitude':    
            bmp = mag_vis(psi_for_vis.T / scale, cmap=self.mag_cmap,  mag_map=self.mag_map)
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
        
        # remove outlines of markers
        mpl.rcParams['scatter.marker'] = 'o'
        mpl.rcParams['scatter.edgecolors'] = 'none'
        
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
        self.anim.ytick_pad = 25 
        
        if isinstance(self.anim, Animator1d):
            # add legend to anim.ax, labels are 'real', 'imat', 'abs', 'pot'
            self.anim.show_legend = True    
        
        
    
    def frame_postprocess(self, anim : AnimatorBase):
        
        return None
        