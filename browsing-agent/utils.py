def parse_tools(tools):
    """Parse tools into a standardized format for function calling.
    
    Args:
        tools: List of tool objects with name, description, and inputSchema
            
    Returns:
        List of tools in the format expected by the LLM client
    """
    formatted_tools = []
    for tool in tools:
        # Handle both dictionary and object access
        name = tool.name if hasattr(tool, 'name') else tool.get('name', '')
        description = getattr(tool, 'description', '') or tool.get('description', '')
        input_schema = getattr(tool, 'inputSchema', None) or tool.get('inputSchema', {})
        
        # Extract parameters from inputSchema
        parameters = {
            'type': 'object',
            'properties': {},
            'required': []
        }
        
        if input_schema and 'properties' in input_schema:
            parameters['properties'] = input_schema['properties']
            
            # Make all properties required
            parameters['required'] = list(input_schema['properties'].keys())
            
            # Add additionalProperties if specified, default to False
            parameters['additionalProperties'] = input_schema.get('additionalProperties', False)
        
        tool_def = {
            'type': 'function',
            'function': {
                'name': name,
                'description': description or '',
                'parameters': parameters,
                'strict': True
            }
        }
        
        formatted_tools.append(tool_def)
    
    return formatted_tools
            