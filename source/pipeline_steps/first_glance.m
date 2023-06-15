function [polygons, roi_points, intens] = first_glance(gui_params_json)
% this version designated to run on the cluster.
% based on clicky3 from AEC and JMK, 12 Oct. 2011.
[gui_time, home_dir, fileName, partial, start_frame, end_frame] = parse_input_params(gui_params_json);
%dt = strcat("Started at: " ,datestr(now,'HH:MM:SS'));
%disp(dt);

[fPath, fName, fExt] = fileparts(fileName);
fileName = [fName, fExt];
switch lower(fExt)
  case '.tif'
	if ~partial
		[movie_in, nframe] = read_tif_video(home_dir, fileName);
	else 
		[movie_in, nframe] = read_tif_video(home_dir, fileName, start_frame, end_frame);
	end
  case '.raw'
	if ~partial
		[movie_in, nframe] = ReadRaw_from_cluster(home_dir, fileName);
	else
		[movie_in, nframe] = ReadRaw_from_cluster(home_dir, fileName, start_frame, end_frame);
	end
  otherwise  % Under all circumstances SWITCH gets an OTHERWISE!
    error('Unexpected file extension: %s', fExt);
end

%disp("Finish to read movie. Starting caculate intensities.");

refimg = mean(movie_in, 3);
nframes = size(movie_in, 3);

figure;
subplot(3,1,1);
imshow(refimg, [], 'InitialMagnification', 'fit');
hold on;

[ysize, xsize] = size(refimg(:,:,1));
npts = 1;
colorindex = 0;
order = get(gca,'ColorOrder');
nroi = 1;
intens = [];
[x, y] = meshgrid(1:xsize, 1:ysize);
polygons = extract_polygons_cluster(home_dir);
number_of_polygons = length(polygons );
for i=1 : number_of_polygons
    subplot(3,1,1);
    poly_points = polygons{1,i};
    xv = poly_points(:,1); yv = poly_points(:,2);
    inpoly = (inpolygon(x,y,xv,yv));

    %draw the bounding polygons and label them
    currcolor = order(1+mod(colorindex,size(order,1)),:);
    plot(xv, yv, 'Linewidth', 1,'Color',currcolor);
    text(mean(xv),mean(yv),num2str(colorindex+1),'Color',currcolor,'FontSize',12);

    itrace = squeeze(sum(sum(movie_in.*repmat(inpoly, [1, 1, nframes]))))/sum(inpoly(:));

    subplot(3,1,2:3); % plot the trace
    hold on;
    plot(itrace,'Color',currcolor);
    colorindex = colorindex+1;
    roi_points{nroi} = [xv, yv];
    nroi = nroi + 1;
	intens = [intens; itrace'];
end
intens = intens';
if ~exist(fullfile(home_dir, 'traces'), 'dir')
   mkdir(fullfile(home_dir, 'traces'))
end
writematrix(intens, fullfile(home_dir,'traces', strcat(gui_time, '_traces.csv')));
if ~exist(fullfile(home_dir, 'Traces_ROIs'), 'dir')
   mkdir(fullfile(home_dir, 'Traces_ROIs'))
end
dt = datestr(now,'yy-mm-dd_HH-MM-SS');
saveas(gca,fullfile(home_dir, 'Traces_ROIs', strcat(gui_time, '.fig')));
finish_task = strcat("Finished at: " ,dt);
%disp(finish_task);
disp("Everything worked well. The Script finished to run."); 	
end

function corrected_polygons = extract_polygons_cluster(home_dir)
% return cell array {1, N}, N = number of polygons (i.e neurons)
% corrected_polygons{1,i} is a duble matrix (P,2), P = number of points that 
% generate this polygong. each point is [x, y] relative to the rectangle
% ROI from ThorImage
dom_dir = home_dir;
while true
	try
		dom = xmlread(fullfile(dom_dir, 'ROIs.xaml'));
		break
	catch
		parts = strsplit(dom_dir, '/');
		parts = parts(~cellfun('isempty',parts));
		l= length(parts);
		dom_dir = '//';
			for i=1:l-1
				dom_dir = strcat(dom_dir,parts{i},'/' );
			end
	end
end
xml_in_struct = xml2structCopy(dom);
% extract polygons of ROIS.
% take only half from them, since they duplicate for blue and red laser
polygons_struct = xml_in_struct.ROICapsule.ROICapsule_dot_ROIs.x_colon_Array.ROIPoly;
number_of_polygons = length(polygons_struct);
poly_lst = {};
for i = 1 :  number_of_polygons 
    points_list = polygons_struct{1,i}.Attributes.Points;
    if ~any(strcmp(poly_lst,points_list))
        poly_lst = [poly_lst; points_list];
    end
end
poly_lst = poly_lst';
%poly_lst = cell(1,number_of_polygons ); % will contain the polygons 
%for i = 1 :  number_of_polygons 
%    points_list = polygons_struct{1,i}.Attributes.Points;
%    poly_lst{i} = points_list;
%end
%poly_lst = unique(poly_lst);
% delete point roi
index_bad = [];
for i = 1 : length(poly_lst)
    temp = poly_lst{i};
    if length(strsplit(temp,','))<= 2
       index_bad = [index_bad,i];
    end
end
poly_lst(index_bad)=[];
number_of_polygons = length(poly_lst);
% extract the coordinates of the rectangle ROI
rect_data = xml_in_struct.ROICapsule.ROICapsule_dot_ROIs.x_colon_Array.ROIRect.Attributes;
bottom_left = strsplit(rect_data.BottomLeft, ','); bottom_left_x = str2double(bottom_left{1,1}); bottom_left_y = str2double(bottom_left{1,2});
top_left = strsplit(rect_data.TopLeft, ','); top_left_x = str2double(top_left{1,1}); top_left_y = str2double(top_left{1,2});
height = str2double(rect_data.ROIHeight);
width = str2double(rect_data.ROIWidth);
% generate list of polygons w.r.t the rectangle ROI
corrected_polygons = cell(1, number_of_polygons);
for i = 1 :  number_of_polygons  % for each polygon
    splitted_points = strsplit(poly_lst{1,i});
    corrected_points = zeros(length(splitted_points) + 1,2);
    for j = 1 :  length(splitted_points) % for each point
        point = strsplit(splitted_points{j}, ',');
        x = str2double(point{1,1}); y = str2double(point{1,2});
        % if the point exceeds the rectangle from above, left or right - trunc it
        x = min(max(x - bottom_left_x, 1),width); 
        y = max(1,min(y - top_left_y, height)) ;
        corrected_points(j,:) = [x, y];
    end
    corrected_points(length(splitted_points) + 1,:) = corrected_points(1,:);
    corrected_polygons{1,i} = corrected_points;
end
end

function [mov, nframe] = ReadRaw_from_cluster(home_dir, fileName, start_frame, end_frame)
% [mov nframe] = readBinMov(fileName, nrow, ncol)
% If only one input: mov = readBinMov(fileName, nrow, ncol)
% The output, mov, is a 3-D array or unsigned 16-bit integers
% The binary input file is assumed to be written by Labview from a 
% Hamamatsu .dcimg file.  The binary file has 16-bit integers with a little
% endian format.
% The frame size information must be independently documented in a .txt 
% file and read into this function as the number of rows (nrow) and the 
% number of columns (ncol).
% read file into tmp vector
xml_dir = home_dir;
while true
	try
		Info = readstruct(fullfile(xml_dir, 'Experiment.xml'));
		break
	catch
		parts = strsplit(xml_dir, '/');
		parts = parts(~cellfun('isempty',parts));
		l= length(parts);
		xml_dir = '//';
			for i=1:l-1
				xml_dir = strcat(xml_dir,parts{i},'/' );
			end
	end
end
nrow = Info.Camera.heightAttribute;
ncol = Info.Camera.widthAttribute;
fid = fopen(fullfile(home_dir, fileName));                  % open file
if nargin == 4
    try
        start_frame = str2num(start_frame);
        end_frame = str2num(end_frame);
    catch
    end
    first_frame  = start_frame *  nrow * ncol;
    nframes  = (end_frame - start_frame) *  nrow * ncol;
    fseek(fid,first_frame,'bof');
    tmp = fread(fid, nframes, '*uint16', 'l');  
end	
if nargin == 2
    tmp = fread(fid,  '*uint16', 'l');       % uint16, little endian
end
fclose(fid);                            % close file

% reshape vector into appropriately oriented, 3D array
L = length(tmp)/(nrow*ncol);
mov = reshape(tmp, [ncol nrow L]);
mov = double(permute(mov, [2 1 3]));

if nargout > 1
    nframe = L;
end
end

function [mov, nframe] = read_tif_video(home_dir, fileName, start_frame, end_frame)
filename = fullfile(home_dir, fileName);
info = imfinfo(filename);
nframe = length(info);
for K = 1 : nframe
    rawframes(:,:,:,K) = imread(filename, K);
end
mov = double(squeeze(rawframes));
if nargin == 4
mov = mov(:,:,start_frame:end_frame)
s = size(mov);
nframe = s(3);
end
end

function [gui_time, home_dir, fileName, partial, start_frame, end_frame] = parse_input_params(gui_params_json)
% Load the JSON data from a file
json_str = fileread(gui_params_json);
% Decode the JSON string into a MATLAB struct
data = jsondecode(json_str);
% Access the data in the struct
gui_time = data.gui_time;
home_dir = data.home_dir_linux;
fileName = data.raw_video_path_linux;
partial = data.partial_video;
start_frame = data.start_frmae_partial;
end_frame = data.end_frmae_partial;
end


function [ s ] = xml2structCopy( file )
%a copy of the exact original function putted here
% due to some non solved bug that doesn't worke without this
%Convert xml file into a MATLAB structure
% [ s ] = xml2struct( file )
%
% A file containing:
% <XMLname attrib1="Some value">
%   <Element>Some text</Element>
%   <DifferentElement attrib2="2">Some more text</Element>
%   <DifferentElement attrib3="2" attrib4="1">Even more text</DifferentElement>
% </XMLname>
%
% Will produce:
% s.XMLname.Attributes.attrib1 = "Some value";
% s.XMLname.Element.Text = "Some text";
% s.XMLname.DifferentElement{1}.Attributes.attrib2 = "2";
% s.XMLname.DifferentElement{1}.Text = "Some more text";
% s.XMLname.DifferentElement{2}.Attributes.attrib3 = "2";
% s.XMLname.DifferentElement{2}.Attributes.attrib4 = "1";
% s.XMLname.DifferentElement{2}.Text = "Even more text";
%
% Please note that the following characters are substituted
% '-' by '_dash_', ':' by '_colon_' and '.' by '_dot_'
%
% Written by W. Falkena, ASTI, TUDelft, 21-08-2010
% Attribute parsing speed increased by 40% by A. Wanner, 14-6-2011
% Added CDATA support by I. Smirnov, 20-3-2012
%
% Modified by X. Mo, University of Wisconsin, 12-5-2012

    if (nargin < 1)
        clc;
        help xml2struct
        return
    end
    
    if isa(file, 'org.apache.xerces.dom.DeferredDocumentImpl') || isa(file, 'org.apache.xerces.dom.DeferredElementImpl')
        % input is a java xml object
        xDoc = file;
    else
        %check for existance
        if (exist(file,'file') == 0)
            %Perhaps the xml extension was omitted from the file name. Add the
            %extension and try again.
            if (isempty(strfind(file,'.xml')))
                file = [file '.xml'];
            end
            
            if (exist(file,'file') == 0)
                error(['The file ' file ' could not be found']);
            end
        end
        %read the xml file
        xDoc = xmlread(file);
    end
    
    %parse xDoc into a MATLAB structure
    s = parseChildNodes(xDoc);
    
end

% ----- Subfunction parseChildNodes -----
function [children,ptext,textflag] = parseChildNodes(theNode)
    % Recurse over node children.
    children = struct;
    ptext = struct; textflag = 'Text';
    if hasChildNodes(theNode)
        childNodes = getChildNodes(theNode);
        numChildNodes = getLength(childNodes);

        for count = 1:numChildNodes
            theChild = item(childNodes,count-1);
            [text,name,attr,childs,textflag] = getNodeData(theChild);
            
            if (~strcmp(name,'#text') && ~strcmp(name,'#comment') && ~strcmp(name,'#cdata_dash_section'))
                %XML allows the same elements to be defined multiple times,
                %put each in a different cell
                if (isfield(children,name))
                    if (~iscell(children.(name)))
                        %put existsing element into cell format
                        children.(name) = {children.(name)};
                    end
                    index = length(children.(name))+1;
                    %add new element
                    children.(name){index} = childs;
                    if(~isempty(fieldnames(text)))
                        children.(name){index} = text; 
                    end
                    if(~isempty(attr)) 
                        children.(name){index}.('Attributes') = attr; 
                    end
                else
                    %add previously unknown (new) element to the structure
                    children.(name) = childs;
                    if(~isempty(text) && ~isempty(fieldnames(text)))
                        children.(name) = text; 
                    end
                    if(~isempty(attr)) 
                        children.(name).('Attributes') = attr; 
                    end
                end
            else
                ptextflag = 'Text';
                if (strcmp(name, '#cdata_dash_section'))
                    ptextflag = 'CDATA';
                elseif (strcmp(name, '#comment'))
                    ptextflag = 'Comment';
                end
                
                %this is the text in an element (i.e., the parentNode) 
                if (~isempty(regexprep(text.(textflag),'[\s]*','')))
                    if (~isfield(ptext,ptextflag) || isempty(ptext.(ptextflag)))
                        ptext.(ptextflag) = text.(textflag);
                    else
                        %what to do when element data is as follows:
                        %<element>Text <!--Comment--> More text</element>
                        
                        %put the text in different cells:
                        % if (~iscell(ptext)) ptext = {ptext}; end
                        % ptext{length(ptext)+1} = text;
                        
                        %just append the text
                        ptext.(ptextflag) = [ptext.(ptextflag) text.(textflag)];
                    end
                end
            end
            
        end
    end
end

% ----- Subfunction getNodeData -----
function [text,name,attr,childs,textflag] = getNodeData(theNode)
    % Create structure of node info.
    
    %make sure name is allowed as structure name
    name = toCharArray(getNodeName(theNode))';
    name = strrep(name, '-', '_dash_');
    name = strrep(name, ':', '_colon_');
    name = strrep(name, '.', '_dot_');

    attr = parseAttributes(theNode);
    if (isempty(fieldnames(attr))) 
        attr = []; 
    end
    
    %parse child nodes
    [childs,text,textflag] = parseChildNodes(theNode);
    
    if (isempty(fieldnames(childs)) && isempty(fieldnames(text)))
        %get the data of any childless nodes
        % faster than if any(strcmp(methods(theNode), 'getData'))
        % no need to try-catch (?)
        % faster than text = char(getData(theNode));
        text.(textflag) = toCharArray(getTextContent(theNode))';
    end
    
end

% ----- Subfunction parseAttributes -----
function attributes = parseAttributes(theNode)
    % Create attributes structure.

    attributes = struct;
    if hasAttributes(theNode)
       theAttributes = getAttributes(theNode);
       numAttributes = getLength(theAttributes);

       for count = 1:numAttributes
            %attrib = item(theAttributes,count-1);
            %attr_name = regexprep(char(getName(attrib)),'[-:.]','_');
            %attributes.(attr_name) = char(getValue(attrib));

            %Suggestion of Adrian Wanner
            str = toCharArray(toString(item(theAttributes,count-1)))';
            k = strfind(str,'='); 
            attr_name = str(1:(k(1)-1));
            attr_name = strrep(attr_name, '-', '_dash_');
            attr_name = strrep(attr_name, ':', '_colon_');
            attr_name = strrep(attr_name, '.', '_dot_');
            attributes.(attr_name) = str((k(1)+2):(end-1));
       end
    end
end