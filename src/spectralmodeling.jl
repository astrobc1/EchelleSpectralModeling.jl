export get_init_parameters, load_templates!, load_template, get_model_λ_grid, get_model_grid_δλ, build

"""
    get_init_parameters
Get the initial parameters.
"""
function get_init_parameters end

"""
    load_templates!
Load and store any templates.
"""
function load_templates! end

"""
    load_template
Load a template
"""
function load_template end

"""
    get_model_λ_grid
Generate the primary high resolution wavelength grid
"""
function get_model_λ_grid end


"""
    get_model_grid_δλ
Generate the primary high resolution wavelength grid spacing (uniform in wavelength)
"""
function get_model_grid_δλ end

"""
    build
Build the model or model component.
"""
function build end