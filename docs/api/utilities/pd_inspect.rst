pd-inspect: Inspect TorchScript Files
=====================================

The ``pd-inspect`` tool inspects tensor fields saved in TorchScript ``.part`` files.

Command Line Usage
------------------

.. code-block:: bash

    pd-inspect <path> [<path> ...]

Arguments:
    * ``path``: Path to ``.part`` file(s) or tar/tar.gz archive(s) containing ``.part`` files

Python API
----------

.. module:: snapy.api.pd_inspect

Functions
~~~~~~~~~

.. function:: inspect_pt_file(path: str, display_name: str = None) -> None

   Load and inspect a single TorchScript .part file.

   :param path: Path to the .part file
   :type path: str
   :param display_name: Optional display name for the file
   :type display_name: str, optional

.. function:: inspect_path(path: str) -> None

   Dispatch based on whether path is a .part file or a tar archive.

   :param path: Path to .part file or tar archive
   :type path: str

.. function:: inspect_script_module(mod: torch.jit.ScriptModule, display_name: str) -> None

   Print information about buffers (tensors) stored in a ScriptModule.

   :param mod: TorchScript module to inspect
   :type mod: torch.jit.ScriptModule
   :param display_name: Name to display for this module
   :type display_name: str

Examples
--------

Inspect a single .part file::

    pd-inspect output.part

Inspect all .part files in a tar archive::

    pd-inspect archive.tar.gz

Using the Python API::

    from snapy.api.pd_inspect import inspect_pt_file

    inspect_pt_file("output.part")

File Format
-----------

Each ``.part`` file is expected to be created by saving tensors as buffers in a TorchScript module:

.. code-block:: python

    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    scripted = torch.jit.script(TensorModule(tensor_map))
    scripted.save(filename)

Output Information
------------------

The tool displays for each tensor:

* Buffer name
* Shape
* First 10 values (for 1D tensors)
* Data type
* Device (CPU/GPU)
* Gradient tracking status
