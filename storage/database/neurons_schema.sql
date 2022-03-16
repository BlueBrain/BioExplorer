create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    name        varchar not null,
    description varchar
);

alter table population
    owner to bioexplorer;

create unique index population_guid_uindex
    on population (guid);

create unique index population_name_uindex
    on population (name);

create table model_template
(
    guid        integer not null
        constraint model_template_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table model_template
    owner to bioexplorer;

create unique index model_template_guid_uindex
    on model_template (guid);

create table node_type
(
    guid        integer not null
        constraint node_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table node_type
    owner to bioexplorer;

create unique index node_type_guid_uindex
    on node_type (guid);

create table model_type
(
    guid        integer not null
        constraint model_type_pk
            primary key,
    code        varchar not null,
    description integer
);

alter table model_type
    owner to bioexplorer;

create unique index model_type_guid_uindex
    on model_type (guid);

create table morphology_type
(
    guid        integer not null
        constraint morphology_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table morphology_type
    owner to bioexplorer;

create unique index morphology_type_guid_uindex
    on morphology_type (guid);

create unique index morphology_type_code_uindex
    on morphology_type (code);

create table configuration
(
    guid  varchar not null
        constraint configuration_pk
            primary key,
    value varchar not null
);

alter table configuration
    owner to bioexplorer;

create unique index configuration_guid_uindex
    on configuration (guid);

create table section_type
(
    guid        integer not null
        constraint section_type_pk
            primary key,
    description varchar not null
);

alter table section_type
    owner to bioexplorer;

create unique index section_type_guid_uindex
    on section_type (guid);

create unique index section_type_description_uindex
    on section_type (description);

create table electrical_type
(
    guid        integer not null
        constraint electrical_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table electrical_type
    owner to bioexplorer;

create unique index electrical_type_guid_uindex
    on electrical_type (guid);

create unique index electrical_type_code_uindex
    on electrical_type (code);

create table morphological_type
(
    guid        integer not null
        constraint morphological_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table morphological_type
    owner to bioexplorer;

create unique index morphological_type_code_uindex
    on morphological_type (code);

create unique index morphological_type_guid_uindex
    on morphological_type (guid);

create table morphology
(
    guid     integer not null
        constraint morphology_pk
            primary key,
    basename varchar not null
);

alter table morphology
    owner to bioexplorer;

create unique index morphology_basename_uindex
    on morphology (basename);

create unique index morphology_guid_uindex
    on morphology (guid);

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
);

alter table section
    owner to bioexplorer;

create unique index section_morphology_guid_section_guid_uindex
    on section (morphology_guid, section_guid);

create index section_morphology_guid_index
    on section (morphology_guid);

create table synapse_connectivity
(
    guid              integer not null,
    presynaptic_guid  integer not null,
    postsynaptic_guid integer not null
        constraint synapse_connectivity_pk
            primary key
);

alter table synapse_connectivity
    owner to bioexplorer;

create unique index synapse_connectivity_guid_uindex
    on synapse_connectivity (guid);

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
);

alter table synapse
    owner to bioexplorer;

create unique index synapse_guid_uindex
    on synapse (guid);

create table layer
(
    guid        integer not null
        constraint layer_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table layer
    owner to bioexplorer;

create unique index layer_guid_uindex
    on layer (guid);

create unique index layer_code_uindex
    on layer (code);

create table morphology_class
(
    guid        integer not null
        constraint morphology_class_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table morphology_class
    owner to bioexplorer;

create unique index morphology_class_guid_uindex
    on morphology_class (guid);

create table synapse_class
(
    guid        integer not null
        constraint synapse_class_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table synapse_class
    owner to bioexplorer;

create unique index synapse_class_guid_uindex
    on synapse_class (guid);

create table region
(
    guid        integer not null
        constraint region_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table region
    owner to bioexplorer;

create unique index region_code_uindex
    on region (code);

create unique index region_guid_uindex
    on region (guid);

create table target
(
    guid        integer not null
        constraint target_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table target
    owner to bioexplorer;

create unique index target_code_uindex
    on target (code);

create unique index target_guid_uindex
    on target (guid);

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
);

alter table node
    owner to bioexplorer;

create unique index node_guid_uindex
    on node (guid);

create table target_node
(
    target_guid integer not null,
    node_guid   integer not null
);

alter table target_node
    owner to bioexplorer;

create index target_node_node_guid_index
    on target_node (node_guid);

create index target_node_target_guid_index
    on target_node (target_guid);

