create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    name        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table population
    owner to bioexplorer;

create unique index population_guid_uindex
    on population (guid)
    tablespace bioexplorer_ts;

create unique index population_name_uindex
    on population (name)
    tablespace bioexplorer_ts;

create table model_template
(
    guid        integer not null
        constraint model_template_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table model_template
    owner to bioexplorer;

create unique index model_template_guid_uindex
    on model_template (guid)
    tablespace bioexplorer_ts;

create table node_type
(
    guid        integer not null
        constraint node_type_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table node_type
    owner to bioexplorer;

create unique index node_type_guid_uindex
    on node_type (guid)
    tablespace bioexplorer_ts;

create table model_type
(
    guid        integer not null
        constraint model_type_pk
            primary key,
    code        varchar not null,
    description integer
)
    tablespace bioexplorer_ts;

alter table model_type
    owner to bioexplorer;

create unique index model_type_guid_uindex
    on model_type (guid)
    tablespace bioexplorer_ts;

create table morphology_type
(
    guid        integer not null
        constraint morphology_type_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table morphology_type
    owner to bioexplorer;

create unique index morphology_type_guid_uindex
    on morphology_type (guid)
    tablespace bioexplorer_ts;

create unique index morphology_type_code_uindex
    on morphology_type (code)
    tablespace bioexplorer_ts;

create table configuration
(
    guid  varchar not null
        constraint configuration_pk
            primary key,
    value varchar not null
)
    tablespace bioexplorer_ts;

alter table configuration
    owner to bioexplorer;

create unique index configuration_guid_uindex
    on configuration (guid)
    tablespace bioexplorer_ts;

create table section_type
(
    guid        integer not null
        constraint section_type_pk
            primary key,
    description varchar not null
)
    tablespace bioexplorer_ts;

alter table section_type
    owner to bioexplorer;

create unique index section_type_guid_uindex
    on section_type (guid)
    tablespace bioexplorer_ts;

create unique index section_type_description_uindex
    on section_type (description)
    tablespace bioexplorer_ts;

create table electrical_type
(
    guid        integer not null
        constraint electrical_type_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table electrical_type
    owner to bioexplorer;

create unique index electrical_type_guid_uindex
    on electrical_type (guid)
    tablespace bioexplorer_ts;

create unique index electrical_type_code_uindex
    on electrical_type (code)
    tablespace bioexplorer_ts;

create table morphological_type
(
    guid        integer not null
        constraint morphological_type_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table morphological_type
    owner to bioexplorer;

create unique index morphological_type_code_uindex
    on morphological_type (code)
    tablespace bioexplorer_ts;

create unique index morphological_type_guid_uindex
    on morphological_type (guid)
    tablespace bioexplorer_ts;

create table morphology
(
    guid     integer not null
        constraint morphology_pk
            primary key,
    basename varchar not null
)
    tablespace bioexplorer_ts;

alter table morphology
    owner to bioexplorer;

create unique index morphology_basename_uindex
    on morphology (basename)
    tablespace bioexplorer_ts;

create unique index morphology_guid_uindex
    on morphology (guid)
    tablespace bioexplorer_ts;

create table section
(
    morphology_guid     integer                    not null,
    section_guid        integer                    not null,
    section_parent_guid integer                    not null,
    section_type_guid   integer                    not null
        constraint section_section_type_guid_fk
            references section_type,
    points              bytea                      not null,
    x                   double precision default 0 not null,
    y                   double precision default 0 not null,
    z                   double precision default 0 not null,
    constraint section_pk
        primary key (morphology_guid, section_guid, section_parent_guid)
)
    tablespace bioexplorer_ts;

alter table section
    owner to bioexplorer;

create unique index section_morphology_guid_section_guid_uindex
    on section (morphology_guid, section_guid)
    tablespace bioexplorer_ts;

create index section_morphology_guid_index
    on section (morphology_guid)
    tablespace bioexplorer_ts;

create table synapse_connectivity
(
    guid              integer not null,
    presynaptic_guid  integer not null,
    postsynaptic_guid integer not null
        constraint synapse_connectivity_pk
            primary key
)
    tablespace bioexplorer_ts;

alter table synapse_connectivity
    owner to bioexplorer;

create unique index synapse_connectivity_guid_uindex
    on synapse_connectivity (guid)
    tablespace bioexplorer_ts;

create table synapse
(
    guid                     integer          not null
        constraint synapse_pk
            primary key,
    presynaptic_neuron_guid  integer          not null,
    postsynaptic_neuron_guid integer          not null,
    surface_x_position       double precision not null,
    surface_y_position       double precision not null,
    surface_z_position       double precision not null,
    center_x_position        double precision not null,
    center_y_position        double precision not null,
    center_z_position        double precision not null
)
    tablespace bioexplorer_ts;

alter table synapse
    owner to bioexplorer;

create unique index synapse_guid_uindex
    on synapse (guid)
    tablespace bioexplorer_ts;

create index synapse_presynaptic_neuron_guid_index
    on synapse (presynaptic_neuron_guid)
    tablespace bioexplorer_ts;

create table layer
(
    guid        integer not null
        constraint layer_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table layer
    owner to bioexplorer;

create unique index layer_guid_uindex
    on layer (guid)
    tablespace bioexplorer_ts;

create unique index layer_code_uindex
    on layer (code)
    tablespace bioexplorer_ts;

create table morphology_class
(
    guid        integer not null
        constraint morphology_class_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table morphology_class
    owner to bioexplorer;

create unique index morphology_class_guid_uindex
    on morphology_class (guid)
    tablespace bioexplorer_ts;

create table synapse_class
(
    guid        integer not null
        constraint synapse_class_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table synapse_class
    owner to bioexplorer;

create unique index synapse_class_guid_uindex
    on synapse_class (guid)
    tablespace bioexplorer_ts;

create table region
(
    guid        integer not null
        constraint region_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table region
    owner to bioexplorer;

create unique index region_code_uindex
    on region (code)
    tablespace bioexplorer_ts;

create unique index region_guid_uindex
    on region (guid)
    tablespace bioexplorer_ts;

create table target
(
    guid        integer not null
        constraint target_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table target
    owner to bioexplorer;

create unique index target_code_uindex
    on target (code)
    tablespace bioexplorer_ts;

create unique index target_guid_uindex
    on target (guid)
    tablespace bioexplorer_ts;

create table node
(
    guid                    integer                    not null
        constraint node_pk
            primary key,
    x                       double precision           not null,
    y                       double precision           not null,
    z                       double precision           not null,
    rotation_x              double precision default 0 not null,
    rotation_y              double precision default 0 not null,
    rotation_z              double precision default 0 not null,
    rotation_w              double precision default 1 not null,
    morphology_guid         integer                    not null,
    morphology_class_guid   integer                    not null,
    electrical_type_guid    integer                    not null,
    morphological_type_guid integer                    not null,
    region_guid             integer                    not null,
    layer_guid              integer                    not null,
    synapse_class_guid      integer                    not null
)
    tablespace bioexplorer_ts;

alter table node
    owner to bioexplorer;

create unique index node_guid_uindex
    on node (guid)
    tablespace bioexplorer_ts;

create table target_node
(
    target_guid integer not null,
    node_guid   integer not null
)
    tablespace bioexplorer_ts;

alter table target_node
    owner to bioexplorer;

create index target_node_node_guid_index
    on target_node (node_guid)
    tablespace bioexplorer_ts;

create index target_node_target_guid_index
    on target_node (target_guid)
    tablespace bioexplorer_ts;

