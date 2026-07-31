Mesh Module
===========

The mesh module provides core functionality for managing computational mesh blocks.

.. module:: snapy

Classes
-------

MeshBlock
~~~~~~~~~

.. class:: MeshBlock

   Mesh block implementation representing a computational block in the domain.

   .. method:: __init__(options: MeshBlockOptions = None)

      Construct a MeshBlock module.

      :param options: Mesh block configuration options
      :type options: MeshBlockOptions, optional

   .. method:: forward(dt: float, stage: int, vars: dict) -> dict

      Forward integration step.

      :param dt: Time step size
      :type dt: float
      :param stage: Integration stage
      :type stage: int
      :param vars: Dictionary of variable tensors
      :type vars: dict[str, torch.Tensor]
      :return: Updated variables dictionary
      :rtype: dict[str, torch.Tensor]

   .. method:: initialize(vars: dict) -> tuple

      Initialize the mesh block.

      :param vars: Variables dictionary
      :type vars: dict
      :return: Tuple of (vars, time)
      :rtype: tuple

   .. method:: max_time_step(vars: dict) -> float

      Calculate maximum stable time step.

      :param vars: Dictionary of variable tensors
      :type vars: dict[str, torch.Tensor]
      :return: Maximum stable time step
      :rtype: float

   .. method:: make_outputs(vars: dict, time: float, final_write: bool = False) -> None

      Generate output files.

      :param vars: Dictionary of variable tensors
      :type vars: dict[str, torch.Tensor]
      :param time: Current simulation time
      :type time: float
      :param final_write: Whether this is a final write
      :type final_write: bool, optional

   .. method:: part(offset: tuple, exterior: bool = True, extend_x1: int = 0, extend_x2: int = 0, extend_x3: int = 0) -> tuple

      Get index slices for a mesh block part.

      :param offset: Index offset tuple (dx3, dx2, dx1)
      :type offset: tuple[int, int, int]
      :param exterior: Whether to include exterior
      :type exterior: bool, optional
      :param extend_x1: Extension in x1 direction
      :type extend_x1: int, optional
      :param extend_x2: Extension in x2 direction
      :type extend_x2: int, optional
      :param extend_x3: Extension in x3 direction
      :type extend_x3: int, optional
      :return: Tuple of slice objects
      :rtype: tuple

   .. method:: device() -> torch.device

      Get the device of the mesh block.

      :return: PyTorch device
      :rtype: torch.device

   .. method:: inc_cycle() -> int

      Increment and return the cycle number.

      :return: Previous cycle number
      :rtype: int

   .. method:: set_user_output_func(func: Callable) -> None

      Set user output callback function.

      :param func: User output function
      :type func: Callable

   .. method:: set_user_stage_forcings(filenames: Sequence[str]) -> None

      Load saved TorchScript forcing modules and apply them sequentially during
      each integration stage.

      Each module must implement ``forward(variables: Dict[str, Tensor],
      dt: float, stage: int) -> Dict[str, Tensor]`` and may return additive
      tendencies keyed by ``"hydro_du"`` and/or ``"scalar_ds"``. The input
      dictionary contains live variables and recursive named buffers from the
      mesh block without copying tensor storage.

      :param filenames: TorchScript ``.pt`` files in execution order
      :type filenames: Sequence[str]

      This method is available on both ``Mesh`` and ``MeshBlock``. Calling it
      on ``Mesh`` loads each file once and shares the modules across all local
      blocks. Their ``forward`` methods must not mutate shared module state.

   .. attribute:: options

      Mesh block configuration options.

      :type: MeshBlockOptions

MeshBlockOptions
~~~~~~~~~~~~~~~~

.. class:: MeshBlockOptions

   Mesh block configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> MeshBlockOptions

      Load MeshBlockOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: MeshBlockOptions loaded from file
      :rtype: MeshBlockOptions

   .. method:: set_bfunc(dx3: int, dx2: int, dx1: int, func: Callable = None) -> None

      Set boundary function for a specific face.

      :param dx3: Direction in x3 (-1, 0, or 1)
      :type dx3: int
      :param dx2: Direction in x2 (-1, 0, or 1)
      :type dx2: int
      :param dx1: Direction in x1 (-1, 0, or 1)
      :type dx1: int
      :param func: Boundary function or None
      :type func: Callable, optional

   .. method:: hydro() -> HydroOptions
               hydro(value: HydroOptions) -> MeshBlockOptions

      Get or set hydro options.

      :return: Hydro options
      :rtype: HydroOptions

   .. method:: layout() -> LayoutOptions
               layout(value: LayoutOptions) -> MeshBlockOptions

      Get or set layout options.

      :return: Layout options
      :rtype: LayoutOptions
